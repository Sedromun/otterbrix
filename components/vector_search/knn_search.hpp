#pragma once

#include "distance_metrics.hpp"
#include "top_k_heap.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace components::vector_search {

    /// Result of a kNN search: row index + distance score.
    using knn_result_t = std::vector<scored_entry_t>;

    /// Perform exact k-Nearest Neighbors search over a flat array of vectors.
    ///
    /// @param data         Pointer to a contiguous array of N vectors, each of `dim` elements,
    ///                     stored row-major: data[i * dim + j] is element j of vector i.
    /// @param num_vectors  Number of vectors (N).
    /// @param dim          Dimensionality of each vector.
    /// @param query        Pointer to the query vector (must have `dim` elements).
    /// @param k            Number of nearest neighbors to return.
    /// @param metric       Distance metric to use.
    /// @returns            Up to `k` results sorted by distance ascending (nearest first).
    template<typename T>
    knn_result_t knn_exact_search(const T* data,
                                  std::size_t num_vectors,
                                  std::size_t dim,
                                  const T* query,
                                  std::size_t k,
                                  metric_type metric) {
        if (dim == 0) {
            throw std::invalid_argument("knn_exact_search: dimension must be > 0");
        }
        if (k == 0) {
            return {};
        }

        top_k_heap_t heap(k);

        for (std::size_t i = 0; i < num_vectors; ++i) {
            const T* vec = data + i * dim;
            double dist = compute_distance(vec, query, dim, metric);
            heap.push(i, dist);
        }

        return heap.drain_sorted();
    }

    /// Convenience overload that accepts std::vector inputs.
    template<typename T>
    knn_result_t knn_exact_search(const std::vector<std::vector<T>>& data,
                                  const std::vector<T>& query,
                                  std::size_t k,
                                  metric_type metric) {
        if (data.empty() || query.empty()) {
            return {};
        }
        std::size_t dim = query.size();

        top_k_heap_t heap(k);

        for (std::size_t i = 0; i < data.size(); ++i) {
            if (data[i].size() != dim) {
                continue; // skip vectors with mismatched dimensionality
            }
            double dist = compute_distance(data[i].data(), query.data(), dim, metric);
            heap.push(i, dist);
        }

        return heap.drain_sorted();
    }

} // namespace components::vector_search
