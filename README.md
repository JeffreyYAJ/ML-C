# YAJ-ML

A Machine Learning library written entirely in **ISO C17**, with no external numerical libraries. YAJ-ML is educational in nature but aims for production-quality code — every algorithm is implemented from scratch so you can understand how classical ML works internally.

## Goals

Implement the following algorithms (in progress):

- Linear Regression
- Logistic Regression
- Perceptron
- k-Nearest Neighbors (KNN)
- Support Vector Machine (Linear SVM)

Future versions may add Decision Trees, Random Forests, Naive Bayes, PCA, and K-Means.

## Requirements

- C17-compatible compiler (GCC, Clang)
- CMake 3.16 or later

No BLAS, LAPACK, OpenCV, Eigen, GSL, or other numerical dependencies.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

For an optimized build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

Or run the test runner directly:

```bash
./build/tests/test_runner
```

## Project Layout

```
include/yaj_ml/   Public headers (error, vector, matrix, model API conventions)
src/              Core library implementation
models/           One directory per ML algorithm (stubs for now)
tests/            Unit tests with a zero-dependency custom harness
examples/         Runnable examples (future)
benchmark/        Performance benchmarks (future)
docs/             Doxygen configuration
```

## Model API Convention

Every model exposes a consistent lifecycle:

```c
<model>_init(...);
<model>_fit(...);
<model>_predict(...);
<model>_score(...);
<model>_free(...);
```

See [`include/yaj_ml/model_api.h`](include/yaj_ml/model_api.h) for ownership rules and naming conventions.

## Documentation

Generate API reference with Doxygen (optional):

```bash
doxygen docs/Doxyfile.in
```

Output is written to `docs/html/`.

## Coding Conventions

- ISO C17, standard library only
- 4-space indentation, K&R brace style
- `snake_case` naming
- Explicit memory ownership (`create`/`free` pairs)
- All public functions return `yaj_ml_status_t`
- Doxygen comments on every public function

## License

See repository for license information.
