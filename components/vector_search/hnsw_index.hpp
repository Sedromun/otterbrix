#pragma once

#include "distance_metrics.hpp"
#include "top_k_heap.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnswlib {
    template<typename T>
    class HierarchicalNSW;
    template<typename T>
    class SpaceInterface;
} // namespace hnswlib

namespace components::vector_search {

    /// Hyperparameters for HNSW construction and search.
    struct hnsw_params_t {
        /// M — number of bi-directional links per node on each layer (typical 8–48).
        std::size_t M = 16;
        /// efConstruction — size of the dynamic candidate list at build time
        /// (typical 100–500). Higher values produce a better-connected graph
        /// at the cost of build time.
        std::size_t ef_construction = 200;
        /// ef_search — size of the dynamic candidate list at query time.
        /// Must be ≥ k. Higher values trade latency for recall.
        std::size_t ef_search = 50;
        /// max_elements — upper bound on the number of indexed points.
        /// HNSW pre-allocates structures of this size.
        std::size_t max_elements = 100000;
        /// Optional random seed for reproducibility.
        std::size_t random_seed = 100;
    };

    /// Approximate kNN index based on Hierarchical Navigable Small World graphs
    /// (Malkov & Yashunin, 2018). Wraps `hnswlib::HierarchicalNSW` with a
    /// thin, type-erased interface independent of the upstream library headers.
    ///
    /// Supported metrics:
    ///   - metric_type::l2            — squared L2 distance
    ///   - metric_type::inner_product — negated dot product
    ///   - metric_type::cosine        — L2 over unit-normalized vectors
    ///                                  (vectors are normalized internally)
    ///
    /// The index is in-memory only. Concurrent insertions are not supported.
    /// Deletes are implemented via `markDelete` in hnswlib but not exposed here.
    class hnsw_index_t {
    public:
        hnsw_index_t(std::size_t dim, metric_type metric, const hnsw_params_t& params = {});

        ~hnsw_index_t();

        hnsw_index_t(const hnsw_index_t&) = delete;
        hnsw_index_t& operator=(const hnsw_index_t&) = delete;
        hnsw_index_t(hnsw_index_t&&) noexcept;
        hnsw_index_t& operator=(hnsw_index_t&&) noexcept;

        /// Add a single point with the given external row identifier.
        /// `vec` must point to exactly `dim()` elements.
        void add(std::size_t row_id, const float* vec);
        void add(std::size_t row_id, const double* vec);

        /// Approximate kNN search. Returns up to `k` results sorted by
        /// ascending distance (nearest first).
        [[nodiscard]] std::vector<scored_entry_t> search(const float* query, std::size_t k) const;
        [[nodiscard]] std::vector<scored_entry_t> search(const double* query, std::size_t k) const;

        /// Adjust ef_search at runtime (controls recall/latency trade-off).
        void set_ef_search(std::size_t ef);

        [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
        [[nodiscard]] metric_type metric() const noexcept { return metric_; }
        [[nodiscard]] std::size_t size() const noexcept;

    private:
        // Normalize a vector into the internal buffer if metric == cosine,
        // otherwise just copy into a float-precision buffer.
        const float* prepare_input(const float* src, std::vector<float>& buf) const;
        const float* prepare_input(const double* src, std::vector<float>& buf) const;

        std::size_t dim_;
        metric_type metric_;
        hnsw_params_t params_;

        std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    };

} // namespace components::vector_search
