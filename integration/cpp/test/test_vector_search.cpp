#include "test_config.hpp"
#include <components/logical_plan/node_vector_search.hpp>
#include <components/logical_plan/param_storage.hpp>
#include <vector_search/distance_metrics.hpp>

#include <catch2/catch.hpp>

#include <cmath>
#include <sstream>
#include <vector>

static const database_name_t database_name = "testdatabase";
static const collection_name_t collection_name = "vectors";

using namespace components;
using namespace components::cursor;

/// Insert a row with a vector column stored as an ARRAY via SQL.
/// Uses: INSERT INTO db.col (id, embedding) VALUES (1, ARRAY[0.1, 0.2, 0.3]);
static std::string make_insert_sql(int id, const std::vector<double>& vec) {
    std::stringstream ss;
    ss << "INSERT INTO TestDatabase.Vectors (id, embedding) VALUES (" << id << ", ARRAY[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << std::fixed << vec[i];
    }
    ss << "]);";
    return ss.str();
}

TEST_CASE("integration::cpp::vector_search::basic_l2") {
    auto config = test_create_config("/tmp/test_vector_search/basic_l2");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    // Setup: create database and collection
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 5 vectors (2D)
    std::vector<std::vector<double>> vectors = {
        {0.0, 0.0}, // id=0, origin
        {1.0, 0.0}, // id=1
        {0.0, 1.0}, // id=2
        {5.0, 5.0}, // id=3, far
        {0.1, 0.1}, // id=4, very close to origin
    };

    for (std::size_t i = 0; i < vectors.size(); ++i) {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(session, make_insert_sql(static_cast<int>(i), vectors[i]));
        REQUIRE(cur->is_success());
    }

    // Verify data inserted
    {
        auto session = otterbrix::session_id_t();
        REQUIRE(dispatcher->size(session, database_name, collection_name) == 5);
    }

    // Vector search: find 3 nearest to origin using L2
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             {0.0, 0.0}, // query = origin
                                             3,
                                             vector_search::metric_type::l2);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 3);
        // The 3 nearest to origin should be: id=0 (d=0), id=4 (d≈0.02), id=1 or id=2 (d=1.0)
    }
}

TEST_CASE("integration::cpp::vector_search::basic_cosine") {
    auto config = test_create_config("/tmp/test_vector_search/basic_cosine");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 4 directional vectors (2D)
    std::vector<std::vector<double>> vectors = {
        {1.0, 0.0},  // id=0: right
        {1.0, 1.0},  // id=1: 45°
        {0.0, 1.0},  // id=2: up
        {-1.0, 0.0}, // id=3: left (opposite)
    };

    for (std::size_t i = 0; i < vectors.size(); ++i) {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(session, make_insert_sql(static_cast<int>(i), vectors[i]));
        REQUIRE(cur->is_success());
    }

    // Search for vectors most similar to "right" direction
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             {1.0, 0.0},
                                             2,
                                             vector_search::metric_type::cosine);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 2);
    }
}

TEST_CASE("integration::cpp::vector_search::k_equals_1") {
    auto config = test_create_config("/tmp/test_vector_search/k_1");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 3 vectors
    for (int i = 0; i < 3; ++i) {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(session, make_insert_sql(i, {static_cast<double>(i), 0.0}));
        REQUIRE(cur->is_success());
    }

    // k=1: nearest to (0,0) should be exactly 1 result
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             {0.0, 0.0},
                                             1,
                                             vector_search::metric_type::l2);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 1);
    }
}

TEST_CASE("integration::cpp::vector_search::k_greater_than_n") {
    auto config = test_create_config("/tmp/test_vector_search/k_gt_n");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 2 vectors
    {
        auto session = otterbrix::session_id_t();
        dispatcher->execute_sql(session, make_insert_sql(0, {1.0, 2.0}));
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->execute_sql(session, make_insert_sql(1, {3.0, 4.0}));
    }

    // k=10 but only 2 rows → should return 2
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             {0.0, 0.0},
                                             10,
                                             vector_search::metric_type::l2);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 2);
    }
}

TEST_CASE("integration::cpp::vector_search::empty_collection") {
    auto config = test_create_config("/tmp/test_vector_search/empty");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Search on empty collection
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             {1.0, 2.0, 3.0},
                                             5,
                                             vector_search::metric_type::l2);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 0);
    }
}

TEST_CASE("integration::cpp::vector_search::higher_dimension") {
    auto config = test_create_config("/tmp/test_vector_search/high_dim");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 10 vectors of dimension 16
    constexpr std::size_t dim = 16;
    constexpr std::size_t n = 10;
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> vec(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            vec[d] = static_cast<double>(i * dim + d) / 100.0;
        }
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(session, make_insert_sql(static_cast<int>(i), vec));
        REQUIRE(cur->is_success());
    }

    // Search k=3
    std::vector<double> query(dim, 0.0);
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->vector_search(session,
                                             database_name,
                                             collection_name,
                                             "embedding",
                                             query,
                                             3,
                                             vector_search::metric_type::cosine);
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() <= 3);
    }
}

TEST_CASE("integration::cpp::vector_search::pre_filtering") {
    auto config = test_create_config("/tmp/test_vector_search/filtered");
    test_clear_directory(config);
    config.disk.on = false;
    config.wal.on = false;
    test_spaces space(config);
    auto* dispatcher = space.dispatcher();

    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_database(session, database_name);
    }
    {
        auto session = otterbrix::session_id_t();
        dispatcher->create_collection(session, database_name, collection_name);
    }

    // Insert 3 documents. Two documents have the same vector, but different category id.
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(
            session,
            "INSERT INTO TestDatabase.Vectors (id, category, embedding) VALUES (1, 10, ARRAY[-1.0, 0.0]);");
        REQUIRE(cur->is_success());
    }
    {
        auto session = otterbrix::session_id_t();
        auto cur = dispatcher->execute_sql(
            session,
            "INSERT INTO TestDatabase.Vectors (id, category, embedding) VALUES (2, 20, ARRAY[1.0, 0.0]);");
        REQUIRE(cur->is_success());
    }
    {
        auto session = otterbrix::session_id_t();
        // Exact same vector as doc 2, but category is 30
        auto cur = dispatcher->execute_sql(
            session,
            "INSERT INTO TestDatabase.Vectors (id, category, embedding) VALUES (3, 30, ARRAY[1.0, 0.0]);");
        REQUIRE(cur->is_success());
    }

    // Query is [1.0, 0.0]. Without filter, doc 2 and 3 should tie for top-1.
    // If we filter by 'category' = 30, only doc 3 should be returned.

    // Construct the filter expression representing: category == 30
    auto filter = components::expressions::make_compare_expression(
        dispatcher->resource(),
        components::expressions::compare_type::eq,
        components::expressions::key_t{dispatcher->resource(), "category"},
        core::parameter_id_t{0});
    // Use the low-level logical plan node constructor via the wrapper directly
    auto node = components::logical_plan::make_node_vector_search(dispatcher->resource(),
                                                                  {database_name, collection_name},
                                                                  "embedding",
                                                                  {1.0, 0.0},
                                                                  1, // k=1
                                                                  vector_search::metric_type::l2,
                                                                  filter);

    {
        auto session = otterbrix::session_id_t();
        auto params_node = components::logical_plan::make_parameter_node(dispatcher->resource());
        params_node->add_parameter(core::parameter_id_t{0}, 30);
        auto cur = dispatcher->execute_plan(session, std::move(node), std::move(params_node));
        REQUIRE(cur->is_success());
        REQUIRE(cur->size() == 1);

        auto doc_id = cur->value(0, 0).value<std::int64_t>();
        REQUIRE((doc_id == 3)); // Must be document 3, not doc 2.
    }
}
