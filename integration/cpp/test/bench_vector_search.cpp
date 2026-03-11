/**
 * bench_vector_search.cpp
 *
 * Benchmark comparing OtterBrix native Top-K against exact vector search (with/without pre-filter).
 * Compares apples-to-apples by applying the same conditions to both Native SQL and Vector Search.
 *
 * Scenarios:
 *   Table 1: 100% Data (No Filter)
 *   Table 2: 50% Data Filter
 *   Table 3: 10% Data Filter
 */

#include "test_config.hpp"
#include <components/expressions/compare_expression.hpp>
#include <components/logical_plan/node_vector_search.hpp>
#include <components/logical_plan/param_storage.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std::chrono;
using namespace components;

static const database_name_t g_db = "benchdb";
static const collection_name_t g_col = "vectors";
constexpr size_t kDim = 128; // vector dimensionality
constexpr size_t kK = 10;    // nearest neighbours / limit

// ─── helpers ─────────────────────────────────────────────────────────────────

static std::mt19937 rng_global(42);

static std::vector<double> random_vector() {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(kDim);
    for (auto& x : v) {
        x = dist(rng_global);
    }
    return v;
}

static std::string make_insert_sql(int row_idx) {
    auto vec = random_vector();
    std::ostringstream ss;
    ss << "INSERT INTO BenchDB.Vectors (_id, category, embedding) VALUES ('"
       << "id_" << row_idx << "', " << (row_idx % 100) << ", ARRAY[";
    for (size_t i = 0; i < kDim; ++i) {
        if (i > 0)
            ss << ", ";
        ss << std::fixed << std::setprecision(6) << vec[i];
    }
    ss << "]);";
    return ss.str();
}

struct bench_result_t {
    double avg_ms;
    double min_ms;
    double max_ms;
};

// Benchmark a vector_search approach
static bench_result_t measure_vector_search(otterbrix::wrapper_dispatcher_t* disp,
                                            const std::vector<double>& query,
                                            expressions::compare_expression_ptr filter,
                                            int64_t filter_val,
                                            int iterations) {
    auto* resource = disp->resource();
    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        auto node = logical_plan::make_node_vector_search(resource,
                                                          {g_db, g_col},
                                                          "embedding",
                                                          query,
                                                          kK,
                                                          vector_search::metric_type::l2,
                                                          filter);
        auto params_node = logical_plan::make_parameter_node(resource);
        if (filter) {
            params_node->add_parameter(core::parameter_id_t{0}, filter_val);
        }

        auto t0 = high_resolution_clock::now();
        auto session = otterbrix::session_id_t();
        auto cur = disp->execute_plan(session, std::move(node), std::move(params_node));
        auto t1 = high_resolution_clock::now();

        if (!cur->is_success()) {
            std::cerr << "vector_search failed (filter_val=" << filter_val << ")\n";
            std::exit(1);
        }
        timings.push_back(duration<double, std::milli>(t1 - t0).count());
    }

    double total = 0;
    double mn = timings[0], mx = timings[0];
    for (double t : timings) {
        total += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }
    return {total / static_cast<double>(timings.size()), mn, mx};
}

// Benchmark the baseline pure SQL top-K approach
static bench_result_t
measure_sql_baseline(otterbrix::wrapper_dispatcher_t* disp, const std::string& sql, int iterations) {
    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        auto t0 = high_resolution_clock::now();
        auto session = otterbrix::session_id_t();
        auto cur = disp->execute_sql(session, sql);
        auto t1 = high_resolution_clock::now();

        if (!cur->is_success()) {
            std::cerr << "SQL query failed: " << sql << "\n";
            std::exit(1);
        }
        timings.push_back(duration<double, std::milli>(t1 - t0).count());
    }

    double total = 0;
    double mn = timings[0], mx = timings[0];
    for (double t : timings) {
        total += t;
        mn = std::min(mn, t);
        mx = std::max(mx, t);
    }
    return {total / static_cast<double>(timings.size()), mn, mx};
}

struct RunRow {
    size_t rows;
    bench_result_t baseline;
    bench_result_t vs;
};

static void print_table(const std::string& title, const std::vector<RunRow>& results) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::left << std::setw(10) << "Rows" << std::setw(14) << "Baseline avg" << std::setw(14)
              << "Baseline min" << std::setw(14) << "RAG avg" << std::setw(14) << "RAG min" << std::setw(12)
              << "Speedup"
              << "\n"
              << std::string(78, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(10) << r.rows << std::fixed << std::setprecision(2) << std::setw(14)
                  << r.baseline.avg_ms << std::setw(14) << r.baseline.min_ms << std::setw(14) << r.vs.avg_ms
                  << std::setw(14) << r.vs.min_ms << std::setprecision(2) << (r.baseline.avg_ms / r.vs.avg_ms) << "x"
                  << "\n";
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n=== Native Top-K vs KNN Vector Search Benchmark ===\n";
    std::cout << "Dimensions: " << kDim << "  |  Limit/K: " << kK << "\n\n";

    const std::vector<size_t> dataset_sizes = {1000, 3000, 6000};
    constexpr int kIterations = 5;

    std::vector<RunRow> results_100;
    std::vector<RunRow> results_50;
    std::vector<RunRow> results_10;

    for (size_t n_rows : dataset_sizes) {
        std::cout << "Preparing dataset with " << n_rows << " rows...\n";

        // Fresh isolated space per dataset size
        auto config = test_create_config("/tmp/bench_vs_native_" + std::to_string(n_rows));
        test_clear_directory(config);
        config.disk.on = false;
        config.wal.on = false;
        config.log.level = log_t::level::off;
        test_spaces space(config);
        auto* disp = space.dispatcher();
        auto* resource = disp->resource();

        // Create DB + collection
        {
            auto s = otterbrix::session_id_t();
            disp->create_database(s, g_db);
        }
        {
            auto s = otterbrix::session_id_t();
            disp->create_collection(s, g_db, g_col);
        }

        // Insert rows via SQL
        for (size_t i = 0; i < n_rows; ++i) {
            auto cur = disp->execute_sql(otterbrix::session_id_t(), make_insert_sql(static_cast<int>(i)));
            if (!cur->is_success()) {
                std::cerr << "Insert failed at row " << i << "\n";
                return 1;
            }
        }

        auto query = random_vector();

        // ─── TABLE 1: 100% Data ───
        {
            std::string sql = "SELECT * FROM BenchDB.Vectors ORDER BY category ASC LIMIT 10;";
            auto r_base = measure_sql_baseline(disp, sql, kIterations);
            auto r_vs = measure_vector_search(disp, query, nullptr, 0, kIterations);
            results_100.push_back({n_rows, r_base, r_vs});
        }

        // ─── TABLE 2: 50% Data ───
        {
            std::string sql = "SELECT * FROM BenchDB.Vectors WHERE category >= 50 ORDER BY category ASC LIMIT 10;";
            auto r_base = measure_sql_baseline(disp, sql, kIterations);

            auto filter = expressions::make_compare_expression(resource,
                                                               expressions::compare_type::gte,
                                                               expressions::key_t{resource, "category"},
                                                               core::parameter_id_t{0});
            auto r_vs = measure_vector_search(disp, query, filter, 50, kIterations);

            results_50.push_back({n_rows, r_base, r_vs});
        }

        // ─── TABLE 3: 10% Data ───
        {
            std::string sql = "SELECT * FROM BenchDB.Vectors WHERE category >= 90 ORDER BY category ASC LIMIT 10;";
            auto r_base = measure_sql_baseline(disp, sql, kIterations);

            auto filter = expressions::make_compare_expression(resource,
                                                               expressions::compare_type::gte,
                                                               expressions::key_t{resource, "category"},
                                                               core::parameter_id_t{0});
            auto r_vs = measure_vector_search(disp, query, filter, 90, kIterations);

            results_10.push_back({n_rows, r_base, r_vs});
        }
    }

    print_table("Таблица 1: без фильтра (100% данных)", results_100);

    print_table("Таблица 2: фильтр ~50% данных", results_50);

    print_table("Таблица 3: фильтр ~10% данных", results_10);

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
