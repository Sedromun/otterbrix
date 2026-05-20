#pragma once

#include "distance_metrics.hpp"
#include "top_k_heap.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace hnswlib {
    template<typename T>
    class HierarchicalNSW;
    template<typename T>
    class SpaceInterface;
} // namespace hnswlib

namespace components::vector_search {

    // HNSW hyperparameters (Malkov & Yashunin, 2018).
    struct hnsw_params_t {
        std::size_t M = 16;
        std::size_t ef_construction = 200;
        std::size_t ef_search = 50;
        std::size_t max_elements = 100000;
        std::size_t random_seed = 100;
    };

    // Approximate kNN index over float vectors. Supports L2, cosine, inner_product.
    class hnsw_index_t {
    public:
        hnsw_index_t(std::size_t dim, metric_type metric, const hnsw_params_t& params = {});
        ~hnsw_index_t();

        hnsw_index_t(const hnsw_index_t&) = delete;
        hnsw_index_t& operator=(const hnsw_index_t&) = delete;
        hnsw_index_t(hnsw_index_t&&) noexcept;
        hnsw_index_t& operator=(hnsw_index_t&&) noexcept;

        void add(std::size_t row_id, const float* vec);
        void add(std::size_t row_id, const double* vec);

        [[nodiscard]] std::vector<scored_entry_t> search(const float* query, std::size_t k) const;
        [[nodiscard]] std::vector<scored_entry_t> search(const double* query, std::size_t k) const;

        void set_ef_search(std::size_t ef);

        [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
        [[nodiscard]] metric_type metric() const noexcept { return metric_; }
        [[nodiscard]] std::size_t size() const noexcept;

    private:
        const float* prepare_input(const float* src, std::vector<float>& buf) const;
        const float* prepare_input(const double* src, std::vector<float>& buf) const;

        std::size_t dim_;
        metric_type metric_;
        hnsw_params_t params_;
        std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    };

} // namespace components::vector_search
