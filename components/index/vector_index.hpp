#pragma once

#include "forward.hpp"
#include "index.hpp"

#include <vector_search/distance_metrics.hpp>
#include <vector_search/hnsw_index.hpp>

#include <memory>
#include <vector>

namespace components::index {

    /// Vector index over a single embedding column, backed by HNSW
    /// (Malkov & Yashunin, 2018). Slots into the existing index_t hierarchy
    /// so that planner / index_engine code does not need a separate registry.
    ///
    /// Key-value methods inherited from index_t (find, lower_bound, etc.)
    /// are no-ops here — vector indexes only respond to knn_search. They
    /// are kept satisfiable so that the rest of the engine treats this as
    /// any other index_t pointer.
    ///
    /// MVCC: hnswlib does not natively support multi-version concurrency.
    /// Inserts go straight into the graph; pending/commit/revert hooks are
    /// stubbed (no-op). Deletes are not implemented in this version.
    class vector_index_t final : public index_t {
    public:
        vector_index_t(std::pmr::memory_resource* resource,
                       std::string name,
                       const keys_base_storage_t& keys,
                       std::size_t dim,
                       vector_search::metric_type metric,
                       vector_search::hnsw_params_t params = {});

        ~vector_index_t() override;

        /// Add a vector with the given row identifier.
        void add_vector(int64_t row_index, const float* vec);
        void add_vector(int64_t row_index, const double* vec);

        /// Approximate kNN search. Overrides index_t::knn_search.
        std::vector<knn_score_t> knn_search(const float* query,
                                            std::size_t dim,
                                            std::size_t k,
                                            vector_search::metric_type metric) const override;

        /// Adjust the runtime recall/latency trade-off (HNSW ef_search).
        void set_ef_search(std::size_t ef);

        [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] vector_search::metric_type metric() const noexcept { return metric_; }

    private:
        // index_t key-value hooks — degenerate implementations for the
        // vector-only index.
        void insert_impl(value_t, index_value_t) final;
        void remove_impl(value_t) final;
        range find_impl(const value_t&) const final;
        range lower_bound_impl(const value_t&) const final;
        range upper_bound_impl(const value_t&) const final;
        iterator cbegin_impl() const final;
        iterator cend_impl() const final;

        void insert_txn_impl(value_t, int64_t, uint64_t) final;
        void mark_delete_impl(value_t, int64_t, uint64_t) final;
        void commit_insert_impl(uint64_t, uint64_t) final;
        void commit_delete_impl(uint64_t, uint64_t) final;
        void revert_insert_impl(uint64_t) final;
        void cleanup_versions_impl(uint64_t) final;
        void for_each_pending_insert_impl(uint64_t,
                                          const std::function<void(const value_t&, int64_t)>&) const final;
        void for_each_pending_delete_impl(uint64_t,
                                          const std::function<void(const value_t&, int64_t)>&) const final;

        void clean_memory_to_new_elements_impl(std::size_t) final;

    private:
        std::size_t dim_;
        vector_search::metric_type metric_;
        std::unique_ptr<vector_search::hnsw_index_t> backend_;
    };

} // namespace components::index
