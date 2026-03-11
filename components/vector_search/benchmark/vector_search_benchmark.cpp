#include <benchmark/benchmark.h>
#include <vector_search/distance_metrics.hpp>
#include <vector_search/knn_search.hpp>

#include <random>
#include <vector>

using namespace components::vector_search;

// Generate random vectors
static std::vector<double> generate_random_data(std::size_t n, std::size_t dim) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> data(n * dim);
    for (auto& v : data) {
        v = dist(rng);
    }
    return data;
}

// ============================================================================
// Distance Metrics Benchmarks
// ============================================================================

static void BM_CosineDistance(benchmark::State& state) {
    std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(2, dim);
    const double* a = data.data();
    const double* b = data.data() + dim;

    for (auto _ : state) {
        double dist = cosine_distance(a, b, dim);
        benchmark::DoNotOptimize(dist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * dim * sizeof(double) * 2));
}
BENCHMARK(BM_CosineDistance)->RangeMultiplier(2)->Range(4, 1024);

static void BM_L2Distance(benchmark::State& state) {
    std::size_t dim = static_cast<std::size_t>(state.range(0));
    auto data = generate_random_data(2, dim);
    const double* a = data.data();
    const double* b = data.data() + dim;

    for (auto _ : state) {
        double dist = l2_distance_squared(a, b, dim);
        benchmark::DoNotOptimize(dist);
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * dim * sizeof(double) * 2));
}
BENCHMARK(BM_L2Distance)->RangeMultiplier(2)->Range(4, 1024);

// ============================================================================
// exact_knn_search Benchmarks
// ============================================================================

static void BM_KnnExactSearch(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));   // Number of vectors
    std::size_t dim = static_cast<std::size_t>(state.range(1)); // Dimension
    std::size_t k = static_cast<std::size_t>(state.range(2));   // K neighbors

    auto data = generate_random_data(n, dim);
    auto query = generate_random_data(1, dim);

    for (auto _ : state) {
        auto results = knn_exact_search(data.data(), n, dim, query.data(), k, metric_type::l2);
        benchmark::DoNotOptimize(results);
    }

    // Total comparisons per iteration: n
    state.counters["throughput(qps)"] = benchmark::Counter(1, benchmark::Counter::kIsRate);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations() * n * dim * sizeof(double)));
}

// Benchmark sweeping N (number of vectors), Dim=128, K=10
BENCHMARK(BM_KnnExactSearch)
    ->Args({1000, 128, 10})
    ->Args({10000, 128, 10})
    ->Args({100000, 128, 10})
    ->Args({1000000, 128, 10});

// Benchmark sweeping Dim (dimensionality), N=100_000, K=10
BENCHMARK(BM_KnnExactSearch)
    ->Args({100000, 16, 10})
    ->Args({100000, 64, 10})
    ->Args({100000, 128, 10})
    ->Args({100000, 512, 10});

// Benchmark sweeping K, N=100_000, Dim=128
BENCHMARK(BM_KnnExactSearch)
    ->Args({100000, 128, 1})
    ->Args({100000, 128, 10})
    ->Args({100000, 128, 100})
    ->Args({100000, 128, 1000});

BENCHMARK_MAIN();
