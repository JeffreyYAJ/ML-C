# Documentation YAJ-ML (français)

Bienvenue ! Cette documentation est conçue pour **apprendre en lisant** : l'architecture du projet, les outils de build (Makefile et CMake), et les algorithmes implémentés dans le code core.

## Par où commencer ?

| Ordre | Document | Contenu |
|-------|----------|---------|
| 1 | [01_architecture.md](01_architecture.md) | Stack technique, arborescence, flux de compilation |
| 2 | [02_makefile.md](02_makefile.md) | Comment fonctionne le Makefile (recommandé si tu débutes) |
| 3 | [03_cmake.md](03_cmake.md) | Comment fonctionne CMake dans ce projet |
| 4 | [04_algorithmes_core.md](04_algorithmes_core.md) | Vecteurs, matrices, opérations linéaires |
| 5 | [05_tests.md](05_tests.md) | Harness de tests et comment en ajouter |

## Commandes rapides

### Avec Makefile (simple)

```bash
make          # compile + affiche où sont les binaires
make test     # compile et lance les 18 tests
make release  # compile optimisé
make clean    # nettoie
```

### Avec CMake (standard industrie)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Les deux systèmes produisent la **même bibliothèque** et les **mêmes tests**. Utilise celui que tu préfères pour apprendre.

## Documentation API (Doxygen)

Les en-têtes dans `include/yaj_ml/` contiennent des commentaires Doxygen sur chaque fonction publique. Pour générer la référence HTML :

```bash
doxygen docs/Doxyfile.in
# → ouvre docs/html/index.html
```
