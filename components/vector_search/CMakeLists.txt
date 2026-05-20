project(vector_search)

# Core kNN primitives are header-only (metrics, top-K heap, exact search).
# The HNSW wrapper is compiled into a small static library so that hnswlib's
# templates do not leak into every translation unit that depends on us.

set(${PROJECT_NAME}_SOURCES
        hnsw_index.cpp
)

add_library(otterbrix_${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES})
add_library(otterbrix::${PROJECT_NAME} ALIAS otterbrix_${PROJECT_NAME})

target_include_directories(
        otterbrix_${PROJECT_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/components
)

target_link_libraries(
        otterbrix_${PROJECT_NAME}
        PRIVATE hnswlib
)

if (DEV_MODE)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif ()
