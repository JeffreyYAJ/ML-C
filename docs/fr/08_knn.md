# k-Nearest Neighbors (KNN)

Algorithme **instance-based** : pas d'apprentissage de poids. Le modèle **mémorise** les données d'entraînement et prédit en cherchant les \(k\) points les plus proches.

## Principe

```
Pour un nouveau point x :
  1. Calculer la distance à chaque point d'entraînement
  2. Sélectionner les k plus proches voisins
  3. Classification → vote majoritaire
     Régression    → moyenne des cibles des voisins
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`include/yaj_ml/knn.h`](../../include/yaj_ml/knn.h) | API publique |
| [`models/knn/knn.c`](../../models/knn/knn.c) | Implémentation |

## API

```c
knn_model_t model;
knn_config_t config;

knn_init(&model);
knn_config_default(&config);
config.k = 3;
config.task = KNN_TASK_CLASSIFICATION;  // ou KNN_TASK_REGRESSION

knn_fit(&model, &config, X, y, n_samples, n_features);
knn_predict(&model, x, n_features, &prediction);
knn_score(&model, X, y, n_samples, n_features, &score);
knn_free(&model);
```

## Distance euclidienne

\[
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=0}^{p-1} (a_i - b_i)^2}
\]

C'est la même métrique que `vec_norm_l2` appliquée à \((\mathbf{a} - \mathbf{b})\). Utilisée pour mesurer la « proximité » entre points.

**Exemple** en 2D : \(d((0,0), (3,4)) = \sqrt{9+16} = 5\)

## Classification : vote majoritaire

Parmi les \(k\) voisins les plus proches, on compte les labels et on choisit le plus fréquent.

**Exemple** avec \(k=3\), voisins les plus proches ayant les labels `[0, 0, 1]` → prédiction **0** (2 votes contre 1).

En cas d'égalité, le label **le plus petit** est choisi (comportement déterministe).

## Régression : moyenne

\[
\hat{y} = \frac{1}{k} \sum_{i \in \text{voisins}} y_i
\]

Avec \(k=1\), KNN régression retourne exactement la cible du point le plus proche.

## Hyperparamètre \(k\)

| \(k\) | Comportement |
|-------|--------------|
| 1 | Très sensible au bruit (overfitting) |
| 3–5 | Bon compromis en pratique |
| grand | Frontières lisses (underfitting) |

Contrainte : \(1 \leq k \leq n_{\text{samples}}\)

## `fit` = mémorisation

Contrairement à la régression linéaire/logistique, `knn_fit` **copie** les données :

```
fit → alloue X_copy, y_copy → stocke dans le modèle
```

Pas de gradient, pas de poids. C'est le **lazy learning**.

## Complexité

| Opération | Complexité |
|-----------|------------|
| `knn_fit` | O(n × p) — copie des données |
| `knn_predict` | O(n × p + n log n) — distances + tri |
| `knn_score` | O(m × (n × p + n log n)) — m échantillons de test |

Naïf mais correct pour l'apprentissage. Optimisations futures possibles : KD-tree, ball tree.

## Score

| Mode | Métrique |
|------|----------|
| Classification | Accuracy |
| Régression | R² |

## Exemple

```c
#include "yaj_ml/knn.h"

const double X[] = {1, 2, 3, 10, 11, 12};
const double y[] = {0, 0, 0,  1,  1,  1};

knn_init(&model);
knn_config_default(&config);
config.k = 3;
knn_fit(&model, &config, X, y, 6, 1);

knn_predict(&model, (double[]){2.0}, 1, &pred);   // → 0
knn_predict(&model, (double[]){11.0}, 1, &pred);  // → 1
```

## Compiler l'exemple

```bash
make
gcc -std=c17 -Iinclude -o examples/knn_demo examples/knn.c \
    build-make/lib/libyaj_ml_knn.a \
    build-make/lib/libyaj_ml.a -lm
./examples/knn_demo
```

## Comparaison avec les autres modèles

| | Régression lin. | Logistique | KNN |
|---|----------------|------------|-----|
| Apprentissage | Calcule des poids | Calcule des poids | Mémorise les données |
| Prédiction | Formule fermée | Sigmoïde | Recherche de voisins |
| Frontière | Linéaire | Linéaire | Non-linéaire |
| Vitesse fit | Rapide | Itératif | Instantané (copie) |
| Vitesse predict | O(p) | O(p) | O(n × p) |

## Limites actuelles

- Distance euclidienne uniquement
- Tri complet à chaque prédiction (pas d'index spatial)
- Pas de pondération par distance (tous les k voisins comptent pareil)

## Prochain modèle

**Perceptron** ou **SVM** — les deux apprennent une frontière linéaire avec des poids.
