#pragma once

#include "node.hpp"

#include <components/expressions/compare_expression.hpp>
#include <string>
#include <vector>
#include <vector_search/distance_metrics.hpp>

namespace components::logical_plan {

    class node_vector_search_t final : public node_t {
    public:
        node_vector_search_t(std::pmr::memory_resource* resource,
                             const collection_full_name_t& collection,
                             std::string column_name,
                             std::vector<double> query_vector,
                             std::size_t k,
                             vector_search::metric_type metric,
                             vector_search::filter_strategy strategy = vector_search::filter_strategy::post_filter);

        const std::string& column_name() const noexcept { return column_name_; }
        const std::vector<double>& query_vector() const noexcept { return query_vector_; }
        std::size_t k() const noexcept { return k_; }
        vector_search::metric_type metric() const noexcept { return metric_; }
        vector_search::filter_strategy strategy() const noexcept { return strategy_; }

    private:
        hash_t hash_impl() const override;
        std::string to_string_impl() const override;

        std::string column_name_;
        std::vector<double> query_vector_;
        std::size_t k_;
        vector_search::metric_type metric_;
        vector_search::filter_strategy strategy_;
    };

    using node_vector_search_ptr = boost::intrusive_ptr<node_vector_search_t>;

    node_vector_search_ptr
    make_node_vector_search(std::pmr::memory_resource* resource,
                            const collection_full_name_t& collection,
                            std::string column_name,
                            std::vector<double> query_vector,
                            std::size_t k,
                            vector_search::metric_type metric,
                            const expressions::compare_expression_ptr& filter = nullptr,
                            vector_search::filter_strategy strategy = vector_search::filter_strategy::post_filter);

} // namespace components::logical_plan
