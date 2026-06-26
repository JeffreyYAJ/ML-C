YAJ-ML Project Scaffold Plan

Current State

The repo at [/home/yaj/Desktop/ML-C](/home/yaj/Desktop/ML-C) is greenfield: only [README.md](/home/yaj/Desktop/ML-C/README.md) (# ML-C) and git metadata. No C source, build system, or tests exist.

Target Layout

ML-C/
├── CMakeLists.txt              # root: project, C17, warnings, subdirs
├── cmake/
│   └── CompilerWarnings.cmake  # reusable -Wall -Wextra -Wpedantic
├── include/yaj_ml/
│   ├── error.h                 # yaj_ml_status_t + helpers
│   ├── types.h                 # shared scalar/index typedefs
│   ├── vector.h                # yaj_ml_vec_t API
│   └── matrix.h                # yaj_ml_mat_t API
├── src/
│   ├── error.c
│   ├── vector.c
│   └── matrix.c
├── models/                     # empty placeholders only
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── perceptron/
│   ├── knn/
│   └── svm/
├── tests/
│   ├── CMakeLists.txt
│   ├── test_main.c             # custom harness entry
│   ├── test_harness.h          # assert macros, runner
│   ├── test_vector.c
│   └── test_matrix.c
├── examples/                   # empty .gitkeep
├── benchmark/                  # empty .gitkeep
└── docs/                       # Doxygen config stub
    └── Doxyfile.in

Each models/*/ directory gets a .gitkeep and a stub CMakeLists.txt with a comment placeholder — no algorithm code.



Architecture Decisions

1. Library layering

flowchart TB
    subgraph future [Future model layer]
        LR[linear_regression]
        LG[logistic_regression]
        P[perceptron]
        K[knn]
        S[svm]
    end
    subgraph core [Core library - yaj_ml]
        M[matrix]
        V[vector]
        E[error]
    end
    LR --> M
    LG --> M
    P --> V
    K --> V
    S --> M
    M --> V
    V --> E





yaj_ml static library — built from src/*.c, public headers in include/yaj_ml/.



Models (future) — each becomes its own static lib (yaj_ml_linear_regression, etc.) linking yaj_ml. Stub dirs now; wired into CMake with empty targets or commented add_subdirectory blocks.



No external deps — ISO C17 + <stdlib.h>, <math.h>, <string.h>, <stddef.h>, <stdint.h>, <stdbool.h> only.

2. Error handling convention

All public functions return yaj_ml_status_t:

typedef enum {
    YAJ_ML_OK = 0,
    YAJ_ML_ERR_NULL_PTR,
    YAJ_ML_ERR_ALLOC,
    YAJ_ML_ERR_DIM,
    YAJ_ML_ERR_INVALID_ARG,
    YAJ_ML_ERR_NOT_FITTED,
} yaj_ml_status_t;





Output parameters passed as pointers; caller owns all out-of-band buffers unless documented otherwise.



yaj_ml_status_str(status) for readable test/log messages.

3. Core data structures (owned, explicit free)







Type



Fields



Ownership





yaj_ml_vec_t



size_t n; double *data;



vec_create / vec_free; caller must not free data directly





yaj_ml_mat_t



size_t rows, cols; double *data;



row-major layout (data[i*cols + j]); mat_create / mat_free

Key operations to implement in scaffold (enough for all 5 future models):

Vector (vector.h / vector.c):





vec_create, vec_free, vec_copy



vec_dot, vec_add, vec_sub, vec_scale



vec_norm_l2, vec_fill, vec_set_zero

Matrix (matrix.h / matrix.c):





mat_create, mat_free, mat_copy



mat_get / mat_set (inline or macro-free accessors)



mat_mul (naive O(n³), correct first)



mat_transpose



mat_vec_mul (matrix × vector)



mat_add_row (augment with bias column — used by linear/logistic regression later)

All functions documented with Doxygen blocks covering purpose, params, return, ownership, and complexity.

4. Future model API contract (header stub only)

Add [include/yaj_ml/model_api.h](/home/yaj/Desktop/ML-C/include/yaj_ml/model_api.h) documenting the naming convention — no generic vtable yet (keeps C simple and educational):

/* Every model MUST expose:
 *   <model>_init(...)
 *   <model>_fit(...)
 *   <model>_predict(...)
 *   <model>_score(...)
 *   <model>_free(...)
 *
 * Each model owns its struct; init allocates, free releases all internals.
 */

This file is documentation-only for now; actual structs come per-model later.



CMake Build System

Root [CMakeLists.txt](/home/yaj/Desktop/ML-C/CMakeLists.txt)





cmake_minimum_required(VERSION 3.16)



project(yaj_ml VERSION 0.1.0 LANGUAGES C)



set(CMAKE_C_STANDARD 17) + CMAKE_C_STANDARD_REQUIRED ON



Include [cmake/CompilerWarnings.cmake](/home/yaj/Desktop/ML-C/cmake/CompilerWarnings.cmake)



add_library(yaj_ml ...) with target_include_directories(yaj_ml PUBLIC include)



enable_testing() + add_subdirectory(tests)



Model subdirs: add stub models/CMakeLists.txt that only creates empty dirs or add_subdirectory with no-op targets

Suggested build commands (document in README)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure

Debug builds: -g -O0. Release: -O2.



Test Harness (zero external deps)

Custom minimal runner in [tests/test_harness.h](/home/yaj/Desktop/ML-C/tests/test_harness.h):





TEST(name) macro registers a void test_##name(void) function



ASSERT_EQ, ASSERT_NEAR (with epsilon), ASSERT_STATUS_OK



test_main.c runs all registered tests, prints pass/fail counts, returns non-zero on failure



Integrated with CTest via add_test(NAME ... COMMAND test_runner)

Initial test coverage (scaffold scope):





Vector: create/free, dot product, norm, add/sub



Matrix: create/free, get/set, multiply small known matrices, mat_vec_mul



Error: status string lookup

No model tests yet.



Documentation Updates





Expand [README.md](/home/yaj/Desktop/ML-C/README.md): rename to YAJ-ML, goals, build/test instructions, directory map, coding conventions summary (link to guidelines).



Add [docs/Doxyfile.in](/home/yaj/Desktop/ML-C/docs/Doxyfile.in) configured for include/yaj_ml/ — generation is optional/manual (doxygen docs/Doxyfile) and not a build dependency.



Coding Conventions (enforced in scaffold)





4-space indent, K&R braces, snake_case



No macros except include guards and test harness registration



const on read-only inputs (const yaj_ml_vec_t *a)



Every public symbol prefixed yaj_ml_ or vec_/mat_ under the namespace headers



Internal helpers marked static in .c files



Out of Scope (this phase)





Any ML algorithm implementation



Dataset I/O / CSV parsing (add when first model needs it)



Benchmarks



CI/GitHub Actions (can be a follow-up)



BLAS-style optimizations



Implementation Order





Create directory tree + .gitkeep files



Add CMake root + compiler warnings module + yaj_ml library target



Implement error.h/c and types.h



Implement vector.h/c with Doxygen docs



Implement matrix.h/c with Doxygen docs



Build custom test harness + vector/matrix unit tests



Add stub model dirs and model_api.h convention doc



Update README and Doxygen stub



Verify: cmake --build clean, all tests pass, zero warnings with -Wall -Wextra -Wpedantic

