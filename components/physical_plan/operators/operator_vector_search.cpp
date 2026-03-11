#include "operator_vector_search.hpp"

#include <components/physical_plan/operators/scan/full_scan.hpp>
#include <components/physical_plan/operators/transformation.hpp>
#include <services/disk/manager_disk.hpp>
#include <vector_search/knn_search.hpp>
#include <vector_search/top_k_heap.hpp>

namespace components::operators {

    operator_vector_search_t::operator_vector_search_t(std::pmr::memory_resource* resource,
                                                       log_t log,
                                                       collection_full_name_t name,
                                                       std::string column_name,
                                                       std::vector<double> query_vector,
                                                       std::size_t k,
                                                       vector_search::metric_type metric,
                                                       const expressions::compare_expression_ptr& filter)
        : read_only_operator_t(resource, log, operator_type::vector_search)
        , name_(std::move(name))
        , column_name_(std::move(column_name))
        , query_vector_(std::move(query_vector))
        , k_(k)
        , metric_(metric)
        , filter_(filter) {}

    void operator_vector_search_t::on_execute_impl(pipeline::context_t* /*pipeline_context*/) {
        if (name_.empty()) {
            return;
        }
        async_wait();
    }

    actor_zeta::unique_future<void> operator_vector_search_t::await_async_and_resume(pipeline::context_t* ctx) {
        if (log_.is_valid()) {
            trace(log(), "operator_vector_search_t::await_async_and_resume on {}", name_.to_string());
        }

        // Step 1: Get column types from storage
        auto [_t, tf] =
            actor_zeta::send(ctx->disk_address, &services::disk::manager_disk_t::storage_types, ctx->session, name_);
        auto types = co_await std::move(tf);

        // Build filter from expression, if present
        std::unique_ptr<components::table::table_filter_t> filter = nullptr;
        if (filter_) {
            filter = transform_predicate(filter_, types, &ctx->parameters);
        }

        // Step 2: Full scan (or filtered scan) to get target data for kNN
        auto [_s, sf] = actor_zeta::send(ctx->disk_address,
                                         &services::disk::manager_disk_t::storage_scan,
                                         ctx->session,
                                         name_,
                                         std::move(filter), // optional pre-filter
                                         -1,                // no limit
                                         ctx->txn);
        auto data = co_await std::move(sf);

        if (!data || data->size() == 0) {
            output_ = make_operator_data(resource_, std::pmr::vector<types::complex_logical_type>{resource_});
            mark_executed();
            co_return;
        }

        // Step 3: Find the target vector column index by name
        auto target_col = data->column_index(column_name_);

        if (target_col == static_cast<size_t>(-1)) {
            // Column not found — set error and return empty
            set_error("vector_search: column '" + column_name_ + "' not found");
            output_ = make_operator_data(resource_, std::pmr::vector<types::complex_logical_type>{resource_});
            mark_executed();
            co_return;
        }

        // Step 4: Extract vectors from the target column and compute kNN
        uint64_t num_rows = data->size();
        std::size_t dim = query_vector_.size();

        vector_search::top_k_heap_t heap(k_);

        for (uint64_t row = 0; row < num_rows; ++row) {
            auto val = data->value(target_col, row);

            // Check that the value is an ARRAY or LIST type
            auto val_type = val.type().type();
            if (val_type != types::logical_type::ARRAY && val_type != types::logical_type::LIST) {
                continue; // skip non-array values
            }

            const auto& children = val.children();
            if (children.size() != dim) {
                continue; // skip dimension mismatch
            }

            // Convert children to double array
            std::vector<double> row_vec(dim);
            bool valid = true;
            for (std::size_t d = 0; d < dim; ++d) {
                auto child_type = children[d].type().type();
                if (child_type == types::logical_type::DOUBLE) {
                    row_vec[d] = children[d].value<double>();
                } else if (child_type == types::logical_type::FLOAT) {
                    row_vec[d] = static_cast<double>(children[d].value<float>());
                } else if (child_type == types::logical_type::INTEGER) {
                    row_vec[d] = static_cast<double>(children[d].value<int32_t>());
                } else if (child_type == types::logical_type::BIGINT) {
                    row_vec[d] = static_cast<double>(children[d].value<int64_t>());
                } else if (child_type == types::logical_type::SMALLINT) {
                    row_vec[d] = static_cast<double>(children[d].value<int16_t>());
                } else if (child_type == types::logical_type::TINYINT) {
                    row_vec[d] = static_cast<double>(children[d].value<int8_t>());
                } else {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }

            double dist = vector_search::compute_distance(row_vec.data(), query_vector_.data(), dim, metric_);
            heap.push(row, dist);
        }

        // Step 5: Build output with Top-K rows + distance score column
        auto results = heap.drain_sorted();

        if (results.empty()) {
            output_ = make_operator_data(resource_, std::pmr::vector<types::complex_logical_type>{resource_});
            mark_executed();
            co_return;
        }

        // Create a new data chunk with only the selected rows
        // Copy column types from original data
        std::pmr::vector<types::complex_logical_type> out_types(resource_);
        for (uint64_t col = 0; col < data->column_count(); ++col) {
            out_types.push_back(data->data[col].type());
        }
        // Add a score column
        auto score_type = types::complex_logical_type(types::logical_type::DOUBLE, "vector_distance");
        out_types.push_back(score_type);

        auto result_chunk =
            std::make_unique<vector::data_chunk_t>(resource_, out_types, static_cast<uint64_t>(results.size()));

        for (std::size_t i = 0; i < results.size(); ++i) {
            auto source_row = static_cast<uint64_t>(results[i].row_id);

            // Copy all original columns
            for (uint64_t col = 0; col < data->column_count(); ++col) {
                result_chunk->data[col].set_value(i, data->value(col, source_row));
            }
            // Set distance score
            result_chunk->data[data->column_count()].set_value(i,
                                                               types::logical_value_t{resource_, results[i].distance});
        }
        result_chunk->set_cardinality(static_cast<uint64_t>(results.size()));

        output_ = make_operator_data(resource_, std::move(*result_chunk));
        mark_executed();
        co_return;
    }

} // namespace components::operators
