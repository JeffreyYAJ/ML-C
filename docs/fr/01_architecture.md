# 01 — Architecture du projet

## Qu'est-ce que YAJ-ML ?

YAJ-ML est une bibliothèque de Machine Learning écrite en **C17 pur**, sans BLAS, LAPACK, Eigen, etc. L'objectif est pédagogique : comprendre comment les algorithmes ML fonctionnent **de l'intérieur**.

## Stack technique

```
┌─────────────────────────────────────────────────────┐
│  Application / Exemple / Test                       │
│  (tests/test_runner, futurs examples/)              │
└──────────────────────┬──────────────────────────────┘
                       │ lie (link)
┌──────────────────────▼──────────────────────────────┐
│  Modèles ML (futur)                                 │
│  linear_regression, logistic_regression, knn, ...   │
└──────────────────────┬──────────────────────────────┘
                       │ lie
┌──────────────────────▼──────────────────────────────┐
│  Core : libyaj_ml.a                                 │
│  error.c  vector.c  matrix.c                        │
└──────────────────────┬──────────────────────────────┘
                       │ utilise
┌──────────────────────▼──────────────────────────────┐
│  Bibliothèque standard C + libm                     │
│  stdlib, string, math (sqrt)                        │
└─────────────────────────────────────────────────────┘
```

### Couches expliquées

1. **Core (`src/` + `include/yaj_ml/`)**  
   Briques de base : gestion d'erreurs, vecteurs, matrices. Tous les futurs modèles s'appuieront dessus.

2. **Modèles (`models/`)**  
   Un répertoire par algorithme. Chaque modèle sera une petite bibliothèque statique qui **lie** `libyaj_ml.a`.

3. **Tests (`tests/`)**  
   Vérifient que le core fonctionne. Pas de framework externe (pas de Unity, Criterion…) : un harness maison dans `test_harness.h`.

4. **Build**  
   Deux façons de compiler le même code :
   - `Makefile` → simple, explicite, idéal pour apprendre
   - `CMakeLists.txt` → standard en entreprise, génère des Makefiles/Ninja automatiquement

## Arborescence des fichiers

```
ML-C/
├── Makefile                 ← build simple (recommandé pour débuter)
├── CMakeLists.txt           ← build CMake (standard industrie)
├── cmake/CompilerWarnings.cmake
├── include/yaj_ml/          ← headers publics (API)
│   ├── error.h              ← codes d'erreur
│   ├── types.h              ← typedefs (double, size_t)
│   ├── vector.h             ← opérations sur vecteurs
│   ├── matrix.h             ← opérations sur matrices
│   └── model_api.h          ← convention API des futurs modèles
├── src/                     ← implémentation du core
│   ├── error.c
│   ├── vector.c
│   └── matrix.c
├── models/                  ← futurs algorithmes ML (stubs)
├── tests/                   ← tests unitaires
├── examples/                ← futurs exemples exécutables
├── benchmark/               ← futurs benchmarks perf
└── docs/                    ← cette documentation
```

## Flux de compilation (concept)

La compilation en C se fait en **deux étapes** :

```
  .c  ──(cc -c)──►  .o  ──(ar / cc)──►  libyaj_ml.a  ou  exécutable
 source            objet               bibliothèque      test_runner
```

1. **Compilation** : chaque `.c` devient un `.o` (code machine, pas encore exécutable seul).
2. **Liaison (link)** : les `.o` sont assemblés en bibliothèque `.a` ou en programme final.

Pour les tests :

```
test_main.c  ─┐
test_vector.c├──► .o files ──► test_runner (exécutable)
              │                      │
libyaj_ml.a  ─┘                      └── lie libm (-lm) pour sqrt()
```

## Convention API des modèles (futur)

Chaque modèle ML exposera les mêmes 5 fonctions :

```c
<model>_init(...);      // prépare la structure
<model>_fit(...);       // entraîne sur les données
<model>_predict(...);   // prédit une sortie
<model>_score(...);     // évalue la performance
<model>_free(...);      // libère la mémoire
```

Voir `include/yaj_ml/model_api.h` pour les règles de propriété mémoire.

## Gestion des erreurs

Toutes les fonctions publiques retournent un `yaj_ml_status_t` :

| Code | Signification |
|------|---------------|
| `YAJ_ML_OK` | Succès |
| `YAJ_ML_ERR_NULL_PTR` | Pointeur NULL passé |
| `YAJ_ML_ERR_ALLOC` | Échec d'allocation (`malloc`/`calloc`) |
| `YAJ_ML_ERR_DIM` | Dimensions incompatibles (ex. produit scalaire de tailles différentes) |
| `YAJ_ML_ERR_INVALID_ARG` | Argument invalide (ex. n=0) |
| `YAJ_ML_ERR_NOT_FITTED` | Modèle pas encore entraîné (futur) |

**Règle** : ne jamais ignorer le code retour. Toujours vérifier `== YAJ_ML_OK`.

## Propriété mémoire

| Objet | Qui alloue | Qui libère |
|-------|-----------|-----------|
| `yaj_ml_vec_t` | `vec_create()` | `vec_free()` |
| `yaj_ml_mat_t` | `mat_create()` | `mat_free()` |
| Scalars de sortie (`double *out`) | l'appelant | l'appelant |

Ne jamais appeler `free()` directement sur `vec->data` ou `mat->data`.
