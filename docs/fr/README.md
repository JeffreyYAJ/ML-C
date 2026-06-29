# Documentation YAJ-ML (français)

Bienvenue dans la documentation pédagogique de **YAJ-ML**. Ces guides expliquent l'architecture du projet, les outils de compilation, la stack technique et les algorithmes implémentés dans le code — le tout pensé pour apprendre en lisant.

## Par où commencer ?

| Ordre | Document | Contenu |
|-------|----------|---------|
| 1 | [Architecture](01_architecture.md) | Structure des dossiers, couches du projet, flux de données |
| 2 | [Makefile](02_makefile.md) | Compilation simple avec `make` (recommandé pour débuter) |
| 3 | [CMake](03_cmake.md) | Système de build alternatif, plus puissant à grande échelle |
| 4 | [Mathématiques de base](04_core_math.md) | Vecteurs, matrices, formules et implémentations |
| 5 | [Stack technique](05_stack.md) | Langage, outils, conventions, tests |
| 6 | [Régression linéaire](06_linear_regression.md) | Premier modèle ML : équations normales + gradient descent |
| 7 | [Régression logistique](07_logistic_regression.md) | Classification binaire, sigmoïde, BCE |

## Compilation rapide

**Avec Makefile (le plus simple) :**

```bash
make          # compile + lance les tests
make release  # compile optimisé
make clean    # nettoie
```

**Avec CMake :**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Documentation API (Doxygen)

Les en-têtes dans `include/yaj_ml/` contiennent des commentaires Doxygen sur chaque fonction publique. Pour générer la référence HTML :

```bash
doxygen docs/Doxyfile.in
# Résultat dans docs/html/index.html
```

## Prochaines étapes

Une fois le scaffold maîtrisé, le premier modèle ML à implémenter sera la **régression linéaire** dans `models/linear_regression/`.
