find_library(RUST_TV_TABLE_FUNCTION_LIBRARY
        NAMES rust_tvtf
        PATHS
        ${CMAKE_CURRENT_LIST_DIR}/../zngur
        REQUIRED)

if(NOT RUST_TV_TABLE_FUNCTION_LIBRARY)
    message(FATAL_ERROR "Could not find rust_tvtf dylib. Please run Rust build first")
endif()
