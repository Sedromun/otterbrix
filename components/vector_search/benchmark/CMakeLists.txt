add_executable(benchmark_vector_search vector_search_benchmark.cpp)

target_link_libraries(
        benchmark_vector_search
        PRIVATE
        otterbrix_vector_search
        benchmark::benchmark
)

# Output it clearly so we can run it
target_compile_features(benchmark_vector_search PRIVATE cxx_std_20)

add_executable(vector_search_custom_bench vector_search_custom_bench.cpp)

target_link_libraries(
        vector_search_custom_bench
        PRIVATE
        otterbrix_vector_search
        cpp_otterbrix
        otterbrix::test_generaty
        otterbrix::log
        otterbrix::collection
        otterbrix::cursor
        otterbrix::session
        otterbrix::dispatcher
        otterbrix::wal
        otterbrix::disk
        otterbrix::locks
        actor-zeta::actor-zeta
        Catch2::Catch2
        spdlog::spdlog
        msgpackc-cxx
        absl::crc32c
        absl::int128
        Boost::boost
        fmt::fmt
        ZLIB::ZLIB
        BZip2::BZip2
)

target_include_directories(
        vector_search_custom_bench
        PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/components
        ${CMAKE_SOURCE_DIR}/integration/cpp
)

target_compile_features(vector_search_custom_bench PRIVATE cxx_std_20)
