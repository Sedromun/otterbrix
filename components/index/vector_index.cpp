#include "vector_index.hpp"

namespace components::index {

    namespace {

        // Sentinel iterator implementation: every empty_iterator equals every
        // other empty_iterator, so cbegin() == cend() for a vector index.
        class empty_iterator_impl_t final : public index_t::iterator_t::iterator_impl_t {
        public:
            static const index_value_t& sentinel() {
                static const index_value_t kSentinel{};
                return kSentinel;
            }

            index_t::iterator::reference value_ref() const override { return sentinel(); }
            iterator_impl_t* next() override { return this; }
            bool equals(const iterator_impl_t*) const override { return true; }
            bool not_equals(const iterator_impl_t*) const override { return false; }
            iterator_impl_t* copy() const override { return new empty_iterator_impl_t(); }
        };

    } // namespace

    vector_index_t::vector_index_t(std::pmr::memory_resource* resource,
                                   std::string name,
                                   const keys_base_storage_t& keys,
                                   std::size_t dim,
                                   vector_search::metric_type metric,
                                   vector_search::hnsw_params_t params)
        : index_t(resource, logical_plan::index_type::vector_hnsw, std::move(name), keys)
        , dim_(dim)
        , metric_(metric)
        , backend_(std::make_unique<vector_search::hnsw_index_t>(dim, metric, params)) {}

    vector_index_t::~vector_index_t() = default;

    void vector_index_t::add_vector(int64_t row_index, const float* vec) {
        backend_->add(static_cast<std::size_t>(row_index), vec);
    }

    void vector_index_t::add_vector(int64_t row_index, const double* vec) {
        backend_->add(static_cast<std::size_t>(row_index), vec);
    }

    std::vector<knn_score_t> vector_index_t::knn_search(const float* query,
                                                        std::size_t dim,
                                                        std::size_t k,
                                                        vector_search::metric_type metric) const {
        // Defensive checks. Callers are expected to pass matching dim/metric;
        // otherwise the result would be meaningless.
        if (dim != dim_ || metric != metric_) {
            return {};
        }
        auto hits = backend_->search(query, k);
        std::vector<knn_score_t> out;
        out.reserve(hits.size());
        for (const auto& h : hits) {
            out.push_back({static_cast<int64_t>(h.row_id), h.distance});
        }
        return out;
    }

    void vector_index_t::set_ef_search(std::size_t ef) { backend_->set_ef_search(ef); }

    std::size_t vector_index_t::size() const noexcept { return backend_->size(); }

    // ---------------------------------------------------------------------
    // index_t key-value hooks — no-op for a vector-only index.
    // ---------------------------------------------------------------------

    void vector_index_t::insert_impl(value_t, index_value_t) {}
    void vector_index_t::remove_impl(value_t) {}

    index_t::range vector_index_t::find_impl(const value_t&) const {
        return {iterator(new empty_iterator_impl_t()), iterator(new empty_iterator_impl_t())};
    }
    index_t::range vector_index_t::lower_bound_impl(const value_t&) const {
        return {iterator(new empty_iterator_impl_t()), iterator(new empty_iterator_impl_t())};
    }
    index_t::range vector_index_t::upper_bound_impl(const value_t&) const {
        return {iterator(new empty_iterator_impl_t()), iterator(new empty_iterator_impl_t())};
    }
    index_t::iterator vector_index_t::cbegin_impl() const {
        return iterator(new empty_iterator_impl_t());
    }
    index_t::iterator vector_index_t::cend_impl() const {
        return iterator(new empty_iterator_impl_t());
    }

    void vector_index_t::insert_txn_impl(value_t, int64_t, uint64_t) {}
    void vector_index_t::mark_delete_impl(value_t, int64_t, uint64_t) {}
    void vector_index_t::commit_insert_impl(uint64_t, uint64_t) {}
    void vector_index_t::commit_delete_impl(uint64_t, uint64_t) {}
    void vector_index_t::revert_insert_impl(uint64_t) {}
    void vector_index_t::cleanup_versions_impl(uint64_t) {}
    void vector_index_t::for_each_pending_insert_impl(uint64_t,
                                                       const std::function<void(const value_t&, int64_t)>&) const {}
    void vector_index_t::for_each_pending_delete_impl(uint64_t,
                                                       const std::function<void(const value_t&, int64_t)>&) const {}

    void vector_index_t::clean_memory_to_new_elements_impl(std::size_t) {}

} // namespace components::index
