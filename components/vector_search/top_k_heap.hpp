#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

namespace components::vector_search {

    /// An entry in the Top-K result set.
    struct scored_entry_t {
        std::size_t row_id;
        double distance;

        /// Max-heap: the entry with largest distance is at the top.
        bool operator<(const scored_entry_t& other) const { return distance < other.distance; }
    };

    /// Fixed-size max-heap for efficient Top-K selection.
    ///
    /// Maintains at most `k` elements. When a new element arrives whose
    /// distance is smaller than the current worst (the heap top), the worst
    /// is evicted and the new element is inserted. This gives O(N log K)
    /// total cost for scanning N elements and keeping the K nearest.
    class top_k_heap_t {
    public:
        explicit top_k_heap_t(std::size_t k)
            : k_(k) {}

        /// Try to insert an entry. Returns true if it was actually inserted.
        bool push(std::size_t row_id, double distance) {
            if (k_ == 0) {
                return false;
            }
            if (heap_.size() < k_) {
                heap_.push({row_id, distance});
                return true;
            }
            // heap is full — only insert if distance is better (smaller) than worst
            if (distance < heap_.top().distance) {
                heap_.pop();
                heap_.push({row_id, distance});
                return true;
            }
            return false;
        }

        /// Returns the current worst (largest) distance in the heap,
        /// or +inf if the heap is empty.
        [[nodiscard]] double worst_distance() const {
            if (heap_.empty()) {
                return std::numeric_limits<double>::infinity();
            }
            return heap_.top().distance;
        }

        /// Returns the number of elements currently in the heap.
        [[nodiscard]] std::size_t size() const { return heap_.size(); }

        /// Returns the maximum capacity.
        [[nodiscard]] std::size_t capacity() const { return k_; }

        /// Returns true when the heap has reached its capacity.
        [[nodiscard]] bool full() const { return heap_.size() >= k_; }

        /// Drain the heap and return results sorted by distance ascending (nearest first).
        [[nodiscard]] std::vector<scored_entry_t> drain_sorted() {
            std::vector<scored_entry_t> results;
            results.reserve(heap_.size());
            while (!heap_.empty()) {
                results.push_back(heap_.top());
                heap_.pop();
            }
            // heap was max-heap → results are in descending order → reverse
            std::reverse(results.begin(), results.end());
            return results;
        }

    private:
        std::size_t k_;
        std::priority_queue<scored_entry_t> heap_;
    };

} // namespace components::vector_search
