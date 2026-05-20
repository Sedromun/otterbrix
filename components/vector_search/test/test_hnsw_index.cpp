// Tests for the HNSW-backed approximate kNN index.
//
// HNSW is approximate, so we don't require recall@k = 1.0 in general. Instead
// we check:
//   1) the index builds and is searchable with reasonable parameters;
//   2) results are sorted by ascending distance;
//   3) recall@k against an independent brute-force ground truth stays within
//      the practically expected range (≥ 0.9 with the default parameters).

#include <catch2/catch.hpp>
#include <vector_search/distance_metrics.hpp>
#include <vector_search/hnsw_index.hpp>
#include <vector_search/knn_search.hpp>

#include <algorithm>
#include <random>
#include <set>
#include <vector>

using namespace components::vector_search;

namespace {

    struct dataset_t {
        std::vector<std::vector<float>> vectors;
        std::vector<float> query;
    };

    dataset_t make_random_dataset(std::size_t n, std::size_t dim, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        dataset_t ds;
        ds.vectors.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            std::vector<float> v(dim);
            for (auto& x : v) x = dist(rng);
            ds.vectors.push_back(std::move(v));
        }
        ds.query.resize(dim);
        for (auto& x : ds.query) x = dist(rng);
        return ds;
    }

    std::set<std::size_t> brute_force_topk(const dataset_t& ds, std::size_t k, metric_type m) {
        std::vector<std::pair<double, std::size_t>> all;
        all.reserve(ds.vectors.size());
        std::size_t dim = ds.query.size();
        for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
            double d = compute_distance(ds.vectors[i].data(), ds.query.data(), dim, m);
            all.emplace_back(d, i);
        }
        std::sort(all.begin(), all.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::set<std::size_t> out;
        for (std::size_t i = 0; i < std::min(k, all.size()); ++i) {
            out.insert(all[i].second);
        }
        return out;
    }

    double recall_at_k(const std::vector<scored_entry_t>& got, const std::set<std::size_t>& truth) {
        if (truth.empty()) return 1.0;
        std::size_t hits = 0;
        for (const auto& e : got) {
            if (truth.count(e.row_id)) ++hits;
        }
        return static_cast<double>(hits) / static_cast<double>(truth.size());
    }

} // namespace

TEST_CASE("hnsw_index::build_and_search::l2") {
    auto ds = make_random_dataset(/*n=*/500, /*dim=*/32, /*seed=*/42);

    hnsw_params_t params;
    params.max_elements = 500;
    params.M = 16;
    params.ef_construction = 200;
    params.ef_search = 64;

    hnsw_index_t index(/*dim=*/32, metric_type::l2, params);
    for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
        index.add(i, ds.vectors[i].data());
    }
    REQUIRE(index.size() == 500);

    auto results = index.search(ds.query.data(), /*k=*/10);
    REQUIRE(results.size() == 10);
    for (std::size_t i = 1; i < results.size(); ++i) {
        REQUIRE(results[i - 1].distance <= results[i].distance);
    }
}

TEST_CASE("hnsw_index::recall_at_k::l2") {
    for (auto dim : {16u, 64u}) {
        auto ds = make_random_dataset(/*n=*/1000, dim, /*seed=*/7);

        hnsw_params_t params;
        params.max_elements = 1000;
        params.M = 16;
        params.ef_construction = 200;
        params.ef_search = 100;

        hnsw_index_t index(dim, metric_type::l2, params);
        for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
            index.add(i, ds.vectors[i].data());
        }
        auto results = index.search(ds.query.data(), /*k=*/10);
        auto truth = brute_force_topk(ds, /*k=*/10, metric_type::l2);

        double recall = recall_at_k(results, truth);
        INFO("dim=" << dim << " recall=" << recall);
        REQUIRE(recall >= 0.9);
    }
}

TEST_CASE("hnsw_index::recall_at_k::cosine") {
    auto ds = make_random_dataset(/*n=*/1000, /*dim=*/64, /*seed=*/11);

    hnsw_params_t params;
    params.max_elements = 1000;
    params.ef_search = 100;

    hnsw_index_t index(/*dim=*/64, metric_type::cosine, params);
    for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
        index.add(i, ds.vectors[i].data());
    }
    auto results = index.search(ds.query.data(), /*k=*/10);
    auto truth = brute_force_topk(ds, /*k=*/10, metric_type::cosine);

    double recall = recall_at_k(results, truth);
    INFO("cosine recall=" << recall);
    REQUIRE(recall >= 0.9);
}

TEST_CASE("hnsw_index::recall_at_k::inner_product") {
    auto ds = make_random_dataset(/*n=*/1000, /*dim=*/64, /*seed=*/99);

    hnsw_params_t params;
    params.max_elements = 1000;
    params.ef_search = 100;

    hnsw_index_t index(/*dim=*/64, metric_type::inner_product, params);
    for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
        index.add(i, ds.vectors[i].data());
    }
    auto results = index.search(ds.query.data(), /*k=*/10);
    auto truth = brute_force_topk(ds, /*k=*/10, metric_type::inner_product);

    double recall = recall_at_k(results, truth);
    INFO("inner_product recall=" << recall);
    REQUIRE(recall >= 0.9);
}

TEST_CASE("hnsw_index::ef_search_affects_recall") {
    auto ds = make_random_dataset(/*n=*/2000, /*dim=*/64, /*seed=*/123);

    hnsw_params_t params;
    params.max_elements = 2000;
    params.M = 8; // smaller graph degree to amplify ef_search effect

    hnsw_index_t index(/*dim=*/64, metric_type::l2, params);
    for (std::size_t i = 0; i < ds.vectors.size(); ++i) {
        index.add(i, ds.vectors[i].data());
    }

    auto truth = brute_force_topk(ds, /*k=*/10, metric_type::l2);

    index.set_ef_search(16);
    double low_recall = recall_at_k(index.search(ds.query.data(), 10), truth);

    index.set_ef_search(256);
    double high_recall = recall_at_k(index.search(ds.query.data(), 10), truth);

    INFO("low_recall(ef=16)=" << low_recall << " high_recall(ef=256)=" << high_recall);
    REQUIRE(high_recall >= low_recall);
}

TEST_CASE("hnsw_index::accepts_double_inputs") {
    auto ds_f = make_random_dataset(/*n=*/200, /*dim=*/16, /*seed=*/3);
    std::vector<std::vector<double>> ds_d;
    ds_d.reserve(ds_f.vectors.size());
    for (const auto& v : ds_f.vectors) {
        std::vector<double> vd(v.begin(), v.end());
        ds_d.push_back(std::move(vd));
    }
    std::vector<double> query_d(ds_f.query.begin(), ds_f.query.end());

    hnsw_params_t params;
    params.max_elements = 200;
    hnsw_index_t index(/*dim=*/16, metric_type::l2, params);
    for (std::size_t i = 0; i < ds_d.size(); ++i) {
        index.add(i, ds_d[i].data());
    }
    auto results = index.search(query_d.data(), /*k=*/5);
    REQUIRE(results.size() == 5);
}
