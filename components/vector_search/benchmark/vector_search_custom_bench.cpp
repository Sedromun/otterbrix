#include "integration/cpp/test/test_config.hpp"
#include <algorithm>
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <chrono>
#include <components/session/session.hpp>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <vector_search/distance_metrics.hpp>

using namespace components;

static const database_name_t db_name = "testdatabase";
static const collection_name_t coll_name = "testcollection";

static std::string vector_to_string(const std::vector<double>& vec) {
    std::stringstream ss;
    ss << "ARRAY[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i] << (i + 1 == vec.size() ? "]" : ", ");
    }
    return ss.str();
}

static std::vector<double> generate_random_vector(std::size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> data(dim);
    for (auto& v : data) v = dist(rng);
    return data;
}

static void fill_database(otterbrix::wrapper_dispatcher_t* dispatcher,
                          components::session::session_id_t session,
                          std::size_t count,
                          std::size_t dim) {
    std::mt19937 rng(42);

    for (std::size_t i = 0; i < count; i += 100) {
        std::stringstream query;
        query << "INSERT INTO " << db_name << "." << coll_name << " (id, category, embedding) VALUES ";
        std::size_t batch_size = std::min<std::size_t>(100, count - i);
        for (std::size_t j = 0; j < batch_size; ++j) {
            auto vec = generate_random_vector(dim, rng);
            // category 1 to 100 for percentage filtering tests
            int category = (rng() % 100) + 1;
            query << "(" << (i + j) << ", " << category << ", " << vector_to_string(vec) << ")";
            if (j + 1 < batch_size)
                query << ", ";
        }
        query << ";";
        auto cur = dispatcher->execute_sql(session, query.str());
        REQUIRE(cur->is_success());
    }
}

TEST_CASE("vector_search_custom_bench") {
    // Configuration
    const std::vector<std::size_t> row_counts = {1000, 3000, 6000};
    const std::size_t dim = 128; // Using 128 dimensions for the benchmark
    const std::size_t k = 10;

    std::mt19937 rng(1337);
    auto query_vec = generate_random_vector(dim, rng);
    std::string q_str = vector_to_string(query_vec);

    for (std::size_t N : row_counts) {
        std::cout << "\n============================================\n";
        std::cout << "Testing N = " << N << " rows\n";
        std::cout << "============================================\n\n";

        // Setup clean DB environment for each Row Count
        auto config = test_create_config("/tmp/custom_bench_vs/base");
        test_clear_directory(config);
        config.disk.on = false;
        config.wal.on = false;
        test_spaces space(config);
        auto* dispatcher = space.dispatcher();
        auto session = components::session::session_id_t();

        // Init Schema and Data
        dispatcher->create_database(session, db_name);
        dispatcher->create_collection(session, db_name, coll_name);
        fill_database(dispatcher, session, N, dim);

        // ====================================================================
        // Table 1: No Filter
        // Baseline: Native SQL ORDER BY id LIMIT K
        // RAG: ORDER BY vec_distance LIMIT K
        // ====================================================================
        std::cout << "---- Table 1: No Filter (~100% data) ----\n";

        {
            std::string q =
                "SELECT * FROM " + db_name + "." + coll_name + " ORDER BY id LIMIT " + std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "Baseline (Native SQL): " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }

        {
            std::string q = "SELECT * FROM " + db_name + "." + coll_name + " ORDER BY vec_distance(embedding, " +
                            q_str + ", 'l2') LIMIT " + std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "RAG (Vector Search)  : " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }

        // ====================================================================
        // Table 2: ~50% Filter
        // Baseline: Native SQL ORDER BY id LIMIT K + WHERE >= 50
        // RAG: ORDER BY vec_distance LIMIT K + WHERE >= 50
        // ====================================================================
        std::cout << "\n---- Table 2: ~50% Filter (category >= 50) ----\n";

        {
            std::string q = "SELECT * FROM " + db_name + "." + coll_name + " WHERE category >= 50 ORDER BY id LIMIT " +
                            std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "Baseline (Native + WHERE): " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }

        {
            std::string q = "SELECT * FROM " + db_name + "." + coll_name +
                            " WHERE category >= 50 ORDER BY vec_distance(embedding, " + q_str + ", 'l2') LIMIT " +
                            std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "RAG (Vec Search + WHERE) : " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }

        // ====================================================================
        // Table 3: ~10% Filter
        // Baseline: Native SQL ORDER BY id LIMIT K + WHERE >= 90
        // RAG: ORDER BY vec_distance LIMIT K + WHERE >= 90
        // ====================================================================
        std::cout << "\n---- Table 3: ~10% Filter (category >= 90) ----\n";

        {
            std::string q = "SELECT * FROM " + db_name + "." + coll_name + " WHERE category >= 90 ORDER BY id LIMIT " +
                            std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "Baseline (Native + WHERE): " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }

        {
            std::string q = "SELECT * FROM " + db_name + "." + coll_name +
                            " WHERE category >= 90 ORDER BY vec_distance(embedding, " + q_str + ", 'l2') LIMIT " +
                            std::to_string(k) + ";";
            auto start = std::chrono::high_resolution_clock::now();
            auto cur = dispatcher->execute_sql(session, q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms = end - start;
            std::cout << "RAG (Vec Search + WHERE) : " << ms.count() << " ms (results: " << cur->size() << ")\n";
        }
    }
}
