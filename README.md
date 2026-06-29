# YAJ-ML

A Machine Learning library written entirely in **ISO C17**, with no external numerical libraries. YAJ-ML is educational in nature but aims for production-quality code — every algorithm is implemented from scratch so you can understand how classical ML works internally.

## Goals

Implement the following algorithms (in progress):

- Linear Regression — **done** (normal equation + gradient descent)
- Logistic Regression — **done** (batch gradient descent, binary classification)
- Perceptron
- k-Nearest Neighbors (KNN)
- Support Vector Machine (Linear SVM)

Future versions may add Decision Trees, Random Forests, Naive Bayes, PCA, and K-Means.

## Requirements

- C17-compatible compiler (GCC, Clang)
- `make` (recommended) or CMake 3.16+

No BLAS, LAPACK, OpenCV, Eigen, GSL, or other numerical dependencies.

## Build

### Option 1 : Makefile (recommended for beginners)

```bash
make          # compile in debug mode + run tests
make release  # optimized build (-O2)
make test     # compile then run tests
make clean    # remove build-make/
make help     # show available targets
```

Output goes to `build-make/` (library: `build-make/lib/libyaj_ml.a`, tests: `build-make/bin/test_runner`).

### Option 2 : CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

For an optimized build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Output goes to `build/`.

## Run Tests

**With Makefile:**

```bash
make test
```

**With CMake:**

```bash
ctest --test-dir build --output-on-failure
```

Or run the test runner directly:

```bash
./build-make/bin/test_runner    # Makefile
./build/tests/test_runner       # CMake
```

## Project Layout

```
include/yaj_ml/   Public headers (error, vector, matrix, model API conventions)
src/              Core library implementation
models/           One directory per ML algorithm (stubs for now)
tests/            Unit tests with a zero-dependency custom harness
examples/         Runnable examples (future)
benchmark/        Performance benchmarks (future)
docs/             Doxygen configuration + French guides
Makefile          Simple build (recommended to start)
CMakeLists.txt    CMake build (for larger projects / CI)
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

### French guides (recommended for learning)

Step-by-step documentation explaining architecture, Makefile, CMake, algorithms, and the tech stack:

| Guide | Topic |
|-------|-------|
| [docs/fr/README.md](docs/fr/README.md) | Index and quick start |
| [docs/fr/01_architecture.md](docs/fr/01_architecture.md) | Project structure and layers |
| [docs/fr/02_makefile.md](docs/fr/02_makefile.md) | Makefile explained line by line |
| [docs/fr/03_cmake.md](docs/fr/03_cmake.md) | CMake for beginners |
| [docs/fr/04_core_math.md](docs/fr/04_core_math.md) | Vector/matrix algorithms and formulas |
| [docs/fr/05_stack.md](docs/fr/05_stack.md) | Full tech stack overview |
| [docs/fr/06_linear_regression.md](docs/fr/06_linear_regression.md) | Linear regression: math, API, both training methods |
| [docs/fr/07_logistic_regression.md](docs/fr/07_logistic_regression.md) | Logistic regression: sigmoid, BCE, classification |

### API reference (Doxygen)

Public headers in `include/yaj_ml/` contain Doxygen comments on every function. Generate HTML reference:

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
