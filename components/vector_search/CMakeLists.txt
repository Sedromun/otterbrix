project(vector_search)

# Header-only library — use INTERFACE target
add_library(otterbrix_${PROJECT_NAME} INTERFACE)
add_library(otterbrix::${PROJECT_NAME} ALIAS otterbrix_${PROJECT_NAME})

target_include_directories(
        otterbrix_${PROJECT_NAME}
        INTERFACE
        ${CMAKE_SOURCE_DIR}/components
)

if (DEV_MODE)
    add_subdirectory(test)
    add_subdirectory(benchmark)
endif ()
