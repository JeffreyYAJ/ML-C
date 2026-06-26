# 03 — Comprendre CMake (dans YAJ-ML)

CMake est un **générateur de build**, pas un compilateur. Il ne compile pas directement : il **écrit** des Makefiles (ou des fichiers Ninja) que tu exécutes ensuite.

```
CMakeLists.txt  ──(cmake)──►  build/Makefile  ──(make/cmake --build)──►  binaires
   (recette)                    (recette générée)                        (résultat)
```

Si le Makefile te semble plus clair, c'est normal : CMake ajoute une couche d'abstraction. Mais c'est le standard dans l'industrie C/C++.

## Les 3 commandes essentielles

```bash
# 1. Configurer (une fois, ou quand CMakeLists.txt change)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug

# 2. Compiler
cmake --build build

# 3. Tester
ctest --test-dir build --output-on-failure
```

| Flag | Signification |
|------|---------------|
| `-S .` | **S**ource : répertoire contenant CMakeLists.txt (racine du projet) |
| `-B build` | **B**uild : répertoire où CMake écrit ses fichiers générés |
| `-DCMAKE_BUILD_TYPE=Debug` | mode debug (-g -O0) |

## CMakeLists.txt racine — ligne par ligne

Fichier : [`CMakeLists.txt`](../../CMakeLists.txt)

```cmake
cmake_minimum_required(VERSION 3.16)
```
Version minimale de CMake requise.

```cmake
project(yaj_ml VERSION 0.1.0 LANGUAGES C)
```
Nom du projet, version, langage C uniquement.

```cmake
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
```
Force le standard **ISO C17** (pas de GNU extensions).

```cmake
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
```
Si tu oublies de préciser Debug/Release, CMake choisit Debug par défaut.

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
```
Charge notre module custom qui active `-Wall -Wextra -Wpedantic`.

```cmake
add_library(yaj_ml STATIC
    src/error.c
    src/vector.c
    src/matrix.c
)
```
Crée une **bibliothèque statique** `libyaj_ml.a` à partir des 3 fichiers source.

```cmake
target_include_directories(yaj_ml PUBLIC include)
```
Quand tu `#include "yaj_ml/vector.h"`, le compilateur cherche dans `include/`.

```cmake
target_link_libraries(yaj_ml PUBLIC m)
```
Lie la bibliothèque math (`libm`) — nécessaire pour `sqrt()` dans `vec_norm_l2`.

```cmake
add_subdirectory(models)
enable_testing()
add_subdirectory(tests)
```
- `models/` : sous-répertoires pour les futurs modèles ML (stubs pour l'instant)
- `enable_testing()` : active CTest
- `tests/` : construit `test_runner` et l'enregistre comme test

## tests/CMakeLists.txt

```cmake
add_executable(test_runner
    test_main.c test_error.c test_vector.c test_matrix.c
)
target_link_libraries(test_runner PRIVATE yaj_ml)
add_test(NAME unit_tests COMMAND test_runner)
```

- `add_executable` : crée un programme (pas une bibliothèque)
- `target_link_libraries(... yaj_ml)` : lie contre `libyaj_ml.a`
- `add_test` : enregistre le test pour `ctest`

## cmake/CompilerWarnings.cmake

```cmake
function(yaj_ml_set_compiler_warnings target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic ...)
endfunction()
```

Une **fonction réutilisable** : on l'appelle sur chaque cible (`yaj_ml`, `test_runner`) pour activer les mêmes warnings partout.

## Ce que CMake génère dans `build/`

```
build/
├── CMakeCache.txt          ← options mémorisées
├── Makefile                ← Makefile généré (tu peux l'ignorer)
├── libyaj_ml.a             ← bibliothèque
└── tests/
    └── test_runner         ← exécutable de tests
```

**Ne modifie jamais** les fichiers dans `build/` à la main. Modifie `CMakeLists.txt` à la racine, puis relance `cmake -S . -B build`.

## Correspondance CMake ↔ Makefile

| Concept | Makefile | CMake |
|---------|----------|-------|
| Compilateur | `CC := gcc` | détecté automatiquement |
| Flags | `CFLAGS := ...` | `target_compile_options(...)` |
| Bibliothèque | `ar rcs libyaj_ml.a ...` | `add_library(yaj_ml STATIC ...)` |
| Exécutable | `gcc -o test_runner ...` | `add_executable(test_runner ...)` |
| Include path | `-Iinclude` | `target_include_directories(...)` |
| Lier libm | `LDFLAGS := -lm` | `target_link_libraries(... m)` |
| Tests | `make test` | `ctest --test-dir build` |

## Quand CMake devient utile

Pour l'instant le projet est petit. CMake devient indispensable quand :
- tu ajoutes 5 modèles ML avec leurs propres bibliothèques
- tu veux supporter Windows (Visual Studio) et Linux
- tu ajoutes des options (`-DYAJ_ML_ENABLE_BENCHMARKS=ON`)
- tu intègres CI/CD (GitHub Actions)

## Workflow recommandé pour apprendre

1. Lis le [Makefile](02_makefile.md) et compile avec `make test`
2. Lis ce document
3. Compile avec CMake et compare les deux dossiers de sortie
4. Modifie un `.c`, recompile avec les deux : observe la recompilation incrémentale
