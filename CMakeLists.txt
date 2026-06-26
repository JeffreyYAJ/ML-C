cmake_minimum_required(VERSION 3.16)

project(yaj_ml VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

add_library(yaj_ml STATIC
    src/error.c
    src/vector.c
    src/matrix.c
)

target_include_directories(yaj_ml
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

yaj_ml_set_compiler_warnings(yaj_ml)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(yaj_ml PRIVATE -g -O0)
else()
    target_compile_options(yaj_ml PRIVATE -O2)
endif()

add_subdirectory(models)
enable_testing()
add_subdirectory(tests)
