# This file is generated by piemaker 0.11.1. DO NOT EDIT this file directly.
cmake_minimum_required(VERSION 3.21)
set(PROJECT_NAME rust-tv-table-function)
set(CMAKE_CXX_STANDARD 17)

if (DEFINED ENV{CMAKE_OSX_ARCHITECTURES})
    set(CMAKE_OSX_ARCHITECTURES $ENV{CMAKE_OSX_ARCHITECTURES})
endif ()

if (DEFINED ENV{ASAN_FLAGS})
    message(STATUS "[build with ASAN] address_sanitizer_flags=$ENV{ASAN_FLAGS}")
    # ENV VAR ASAN_FLAGS could be something like '-fsanitize=address'
    # -g to print line numbers if debug symbols are present
    add_compile_options($ENV{ASAN_FLAGS} -g -fno-omit-frame-pointer)
    add_link_options($ENV{ASAN_FLAGS})
endif ()

message(STATUS "#################### PROJECT ${PROJECT_NAME} ####################")

# https://vcpkg.readthedocs.io/en/latest/users/integration/#cmake
if (DEFINED ENV{VCPKG_ROOT})
    message(STATUS "vcpkg found VCPKG_ROOT=$ENV{VCPKG_ROOT}")
    # https://github.com/microsoft/vcpkg/blob/master/docs/users/integration.md#changing-the-triplet-1

    if (DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
        message(STATUS "using VCPKG_DEFAULT_TRIPLET from environment $ENV{VCPKG_DEFAULT_TRIPLET}")
        set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}")
    endif ()

    if (DEFINED ENV{VCPKG_TARGET_TRIPLET})
        message(STATUS "using VCPKG_TARGET_TRIPLET from environment $ENV{VCPKG_TARGET_TRIPLET}")
        set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_TARGET_TRIPLET}")
    endif ()

    if (DEFINED VCPKG_TARGET_TRIPLET)
        message(STATUS "build with triplet $ENV{VCPKG_ROOT}/triplets/${VCPKG_TARGET_TRIPLET}.cmake")
        include($ENV{VCPKG_ROOT}/triplets/${VCPKG_TARGET_TRIPLET}.cmake)
    endif ()

    if (DEFINED VCPKG_C_FLAGS)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VCPKG_C_FLAGS}")
    endif ()

    if (DEFINED VCPKG_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VCPKG_CXX_FLAGS}")
    endif ()

    if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    endif ()
endif ()

option(ENABLE_TIME_TRACE "enable the time trace for compilation speed profiling" OFF)
if (DEFINED ENV{ENABLE_TIME_TRACE} OR ENABLE_TIME_TRACE)
    message(STATUS "enable -ftime-trace for compilation profiling")

    if (NOT (${CMAKE_CXX_FLAGS} MATCHES "-ftime-trace"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-trace")
    endif ()
endif ()

project(${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()

if (NOT CMAKE_VERBOSE_BUILD)
    set(CMAKE_VERBOSE_BUILD "OFF")
endif ()

set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

###########################################
# project directory layout and conventions
###########################################
set(PROJECT_CPP_SOURCES_DIR src)
set(PROJECT_GENERATED_SOURCES_DIR generated_src)
set(PROJECT_TESTS_DIR tests)
set(PROJECT_CMAKE_DIR cmake)
set(PROJECT_BENCHMARKS_DIR benchmarks)
set(VCPKG_PACKAGE_NAME ${PROJECT_NAME})
set(EXPORT_TARGET_NAME ${PROJECT_NAME}-targets)


function(build_dir BUILD_DIR_RETURN_VALUE)
    string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
    if (APPLE)
        set(BUILD_PLATFORM "osx")
    else ()
        set(BUILD_PLATFORM "linux")
    endif ()
    set(${BUILD_DIR_RETURN_VALUE} build-cmake-${BUILD_TYPE}-${BUILD_PLATFORM} PARENT_SCOPE)
endfunction()

########### modules ###############
function(module_binary_dir module RETURN_VALUE)
    build_dir(BUILD_DIR_RETURN_VALUE)
    set(${RETURN_VALUE} ${module}/${BUILD_DIR_RETURN_VALUE} PARENT_SCOPE)
endfunction()

########### setup dependencies #############
macro(setup_dependencies)
    # add third party dependencies
    # only the top level project will search all dependencies for itself and all its sub projects so that cmake configuration faster

#    find_package(Arrow REQUIRED)

    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        find_library(COREFOUNDATION_LIBRARY NAMES CoreFoundation REQUIRED)
        if (COREFOUNDATION_LIBRARY)
            message(STATUS "[library found] library=CoreFoundation path=${COREFOUNDATION_LIBRARY}")
        else ()
            message(FATAL_ERROR "[library not found] library='CoreFoundation'")
        endif ()

    endif ()
#    find_path(BOOST_ALGORITHM_STRING_HPP_INCLUDE_DIR
#            boost/algorithm/string.hpp
#            REQUIRED)
    message(STATUS "[header directory found] include=boost/algorithm/string.hpp include_dir=${BOOST_ALGORITHM_STRING_HPP_INCLUDE_DIR}")

endmacro()
setup_dependencies()

# add dependent cmake modules
# only add the sub directory when it is not added elsewhere
# most of the cases, using the TARGET like `namespace::name` is sufficient
# but some module may be lack of src folder which means there is no automatic target generated 
# at the same time, cmake doesn't support using `:` in target name yet,
# so we convert the module name's `:` into `_` and use it as a guard as well
########### src #############
function(setup_src)
    set(TARGET_NAME ${PROJECT_NAME})

    add_library(${TARGET_NAME})
    set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)


    # extension for processing resources phase
    # usually this phase generates some source code
    message(STATUS "[processing resources] cmake=${PROJECT_CMAKE_DIR}/process_resources.cmake")
    include(${PROJECT_CMAKE_DIR}/process_resources.cmake)


    # *.h/*.hpp/*.cpp/*.cc will be considered as sources
    file(GLOB_RECURSE TARGET_SOURCES CONFIGURE_DEPENDS
            ${PROJECT_CPP_SOURCES_DIR}/*.h
            ${PROJECT_CPP_SOURCES_DIR}/*.hpp
            ${PROJECT_CPP_SOURCES_DIR}/*.cpp
            ${PROJECT_CPP_SOURCES_DIR}/*.cc
            ${PROJECT_GENERATED_SOURCES_DIR}/*.h
            ${PROJECT_GENERATED_SOURCES_DIR}/*.hpp
            ${PROJECT_GENERATED_SOURCES_DIR}/*.cpp
            ${PROJECT_GENERATED_SOURCES_DIR}/*.cc)
    if (${CMAKE_VERBOSE_BUILD})
        message(STATUS "[glob target sources] target=${TARGET_NAME} generated_source_list=${GENERATED_CPP_SOURCES} source_list=${TARGET_SOURCES}")
    else ()
        list(LENGTH GENERATED_CPP_SOURCES GENERATED_CPP_FILES_COUNT)
        list(LENGTH TARGET_SOURCES CPP_FILES_COUNT)
        message(STATUS "[glob target sources] target=${TARGET_NAME} generated_source_list=${GENERATED_CPP_FILES_COUNT} source_list=${CPP_FILES_COUNT}")
    endif ()
    # file glob above cannot have a dependent target so is not guaranteed to happen before code generation target
    # so project may set the GENERATED_CPP_SOURCES variable during process_resources stage to specify the generated sources
    target_sources(${TARGET_NAME}
            PRIVATE ${TARGET_SOURCES} ${GENERATED_CPP_SOURCES})

    set(PUBLIC_LINKED_LIBRARIES "")
    set(INTERFACE_LINKED_LIBRARIES "")
    set(PRIVATE_LINKED_LIBRARIES "")
#    list(APPEND PRIVATE_LINKED_LIBRARIES Arrow::arrow_static)

    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        list(APPEND PUBLIC_LINKED_LIBRARIES ${COREFOUNDATION_LIBRARY})
    endif ()
    list(APPEND PUBLIC_LINKED_LIBRARIES ${RUST_TV_TABLE_FUNCTION_LIBRARY})

    # for source with only header files, there is no need to link
    # https://cmake.org/cmake/help/v3.15/manual/cmake-buildsystem.7.html#transitive-usage-requirements
    target_link_libraries(${TARGET_NAME}
            PUBLIC
            ${PUBLIC_LINKED_LIBRARIES}
            INTERFACE
            ${INTERFACE_LINKED_LIBRARIES}
            PRIVATE
            ${PRIVATE_LINKED_LIBRARIES}
    )


    target_include_directories(${TARGET_NAME}
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/${PROJECT_CPP_SOURCES_DIR}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/${PROJECT_GENERATED_SOURCES_DIR}>
            $<INSTALL_INTERFACE:include>
    )

    set_property(TARGET ${TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)


    message(STATUS "Install targets for package ${VCPKG_PACKAGE_NAME} into ${CMAKE_INSTALL_LIBDIR}/cmake/${VCPKG_PACKAGE_NAME}")
    # append target export information to the export target name
    install(TARGETS ${PROJECT_NAME}
            EXPORT ${EXPORT_TARGET_NAME}
    )
    # export target information in export target name as a file, and save it into install destination
    install(EXPORT
            ${EXPORT_TARGET_NAME}
            NAMESPACE stonewave::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${VCPKG_PACKAGE_NAME}"
    )

    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in)
        message(STATUS "Installing cmake package config file as ${CMAKE_INSTALL_LIBDIR}/cmake/${VCPKG_PACKAGE_NAME}/${PROJECT_NAME}Config.cmake")
        include(CMakePackageConfigHelpers)
        configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${VCPKG_PACKAGE_NAME}
        )
        install(FILES
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${VCPKG_PACKAGE_NAME}
        )
    else ()
        message(STATUS "No cmake package config file found at ${CMAKE_CURRENT_LIST_DIR}/Config.cmake.in")
    endif ()

endfunction()
setup_src()


########### tests #############
function(setup_tests)
    find_package(Catch2 REQUIRED)

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)

    set(TARGET_NAME rust-tv-table-function-tests)

    add_executable(${TARGET_NAME})


    file(GLOB_RECURSE TARGET_SOURCES CONFIGURE_DEPENDS ${PROJECT_TESTS_DIR}/*.h ${PROJECT_TESTS_DIR}/*.cpp)
    target_sources(${TARGET_NAME}
            PRIVATE ${TARGET_SOURCES})

    target_link_libraries(${TARGET_NAME}
            PRIVATE
            ${PROJECT_NAME}

            Catch2::Catch2
    )
    target_include_directories(${TARGET_NAME}
            PRIVATE
            ${PROJECT_TESTS_DIR}
    )
    include(CTest)
    include(Catch)


    catch_discover_tests(${TARGET_NAME}

            PROPERTIES LABELS basic_test
            OUTPUT_DIR junit
            REPORTER junit
            OUTPUT_SUFFIX .xml
    )
    # copy test data during build time
    # file will only be copied when build succeeds (POST_BUILD)
    # a folder with names like `tests/data` its contents will be copied into `data` in bin folder 


endfunction()

enable_testing()
setup_tests()

message(STATUS "#################### END OF PROJECT ${PROJECT_NAME} ####################") 
