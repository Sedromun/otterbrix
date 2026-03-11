#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace components::vector_search {

    enum class metric_type : uint8_t
    {
        cosine,
        l2
    };

    inline metric_type metric_from_string(const std::string& s) {
        if (s == "cosine" || s == "cos") {
            return metric_type::cosine;
        }
        if (s == "l2" || s == "euclidean") {
            return metric_type::l2;
        }
        throw std::invalid_argument("unknown metric type: " + s);
    }

    inline std::string metric_to_string(metric_type m) {
        switch (m) {
            case metric_type::cosine:
                return "cosine";
            case metric_type::l2:
                return "l2";
        }
        return "unknown";
    }

    /// Compute cosine distance = 1.0 - cosine_similarity.
    /// Returns 0.0 for identical directions, 1.0 for orthogonal, 2.0 for opposite.
    /// If either vector has zero magnitude, returns 1.0 (undefined similarity → max distance).
    template<typename T>
    double cosine_distance(const T* a, const T* b, std::size_t dim) {
        double dot = 0.0;
        double norm_a = 0.0;
        double norm_b = 0.0;

        for (std::size_t i = 0; i < dim; ++i) {
            auto va = static_cast<double>(a[i]);
            auto vb = static_cast<double>(b[i]);
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        constexpr double eps = std::numeric_limits<double>::epsilon();
        if (norm_a <= eps || norm_b <= eps) {
            return 1.0; // undefined similarity
        }

        double similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        // Clamp to [-1, 1] to handle floating point errors
        if (similarity > 1.0) {
            similarity = 1.0;
        }
        if (similarity < -1.0) {
            similarity = -1.0;
        }
        return 1.0 - similarity;
    }

    /// Compute squared L2 (Euclidean) distance.
    /// We use squared distance to avoid the sqrt for performance in comparisons.
    /// The ordering is preserved: if ||a-b||^2 < ||a-c||^2 then ||a-b|| < ||a-c||.
    template<typename T>
    double l2_distance_squared(const T* a, const T* b, std::size_t dim) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += diff * diff;
        }
        return sum;
    }

    /// Compute L2 (Euclidean) distance.
    template<typename T>
    double l2_distance(const T* a, const T* b, std::size_t dim) {
        return std::sqrt(l2_distance_squared(a, b, dim));
    }

    /// Compute distance between two vectors using the specified metric.
    /// Uses squared L2 for metric_type::l2 (preserves ordering, avoids sqrt).
    template<typename T>
    double compute_distance(const T* a, const T* b, std::size_t dim, metric_type metric) {
        switch (metric) {
            case metric_type::cosine:
                return cosine_distance(a, b, dim);
            case metric_type::l2:
                return l2_distance_squared(a, b, dim);
        }
        return 0.0;
    }

} // namespace components::vector_search
