# Régression linéaire

Premier modèle ML de YAJ-ML. Il prédit une valeur continue \(\hat{y}\) à partir de features \(\mathbf{x}\).

## Modèle mathématique

\[
\hat{y} = w_0 x_0 + w_1 x_1 + \cdots + w_{p-1} x_{p-1} + b
\]

- \(p\) = nombre de features (`n_features`)
- \(w_0, \ldots, w_{p-1}\) = coefficients (`weights[0..p-1]`)
- \(b\) = biais / intercept (`weights[p]`)

En notation matricielle avec matrice augmentée \(X_{aug}\) (colonne de 1 ajoutée) :

\[
\hat{\mathbf{y}} = X_{aug} \mathbf{w}
\]

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`include/yaj_ml/linear_regression.h`](../../include/yaj_ml/linear_regression.h) | API publique |
| [`models/linear_regression/linear_regression.c`](../../models/linear_regression/linear_regression.c) | Implémentation |

## API

```c
lr_model_t model;
lr_config_t config;

lr_init(&model, n_features);
lr_config_default(LR_METHOD_NORMAL_EQUATION, &config);
lr_fit(&model, &config, X, y, n_samples, n_features);
lr_predict(&model, x, n_features, &prediction);
lr_score(&model, X, y, n_samples, n_features, &r2);
lr_free(&model);
```

### Format des données

- `X` : matrice **row-major** aplatie — `X[i * n_features + j]` = feature `j` de l'échantillon `i`
- `y` : tableau de `n_samples` valeurs cibles

## Méthode 1 : Équations normales

### Formule

La solution des moindres carrés (MSE minimum) en forme fermée :

\[
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
\]

### Algorithme implémenté

Plutôt que d'inverser explicitement \(X^T X\) (instable numériquement), on **résout le système linéaire** :

\[
(X^T X) \mathbf{w} = X^T \mathbf{y}
\]

Étapes dans `lr_fit_normal_equation()` :

1. `mat_create_from_buffer` — construire la matrice \(X\)
2. `mat_add_row` — augmenter avec la colonne de 1 → \(X_{aug}\)
3. `mat_transpose` — calculer \(X^T\)
4. `mat_mul` — calculer \(X^T X\)
5. `mat_vec_mul` — calculer \(X^T \mathbf{y}\)
6. `mat_solve` — résoudre pour \(\mathbf{w}\) (élimination de Gauss avec pivot partiel)

### Quand l'utiliser

| Avantage | Inconvénient |
|----------|--------------|
| Solution exacte en une passe | O(\(p^3\)) — lent si beaucoup de features |
| Pas de hyperparamètres | \(X^T X\) peut être singulière (features colinéaires) |
| Pédagogique (algèbre linéaire pure) | Coûteux en mémoire pour grands datasets |

## Méthode 2 : Gradient Descent

### Fonction de coût (MSE)

\[
L(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
= \frac{1}{n} \|X_{aug}\mathbf{w} - \mathbf{y}\|^2
\]

### Gradient

\[
\nabla L = \frac{2}{n} X_{aug}^T (X_{aug}\mathbf{w} - \mathbf{y})
\]

### Mise à jour

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla L
\]

où \(\eta\) = `learning_rate`.

### Algorithme implémenté

Boucle dans `lr_fit_gradient_descent()` :

```
initialiser w = 0
pour iter = 1 .. max_iters :
    y_hat = X_aug * w
    residual = y_hat - y
    gradient = (2/n) * X_aug^T * residual
    w = w - learning_rate * gradient
    si ||gradient|| < tolerance : arrêter
```

### Hyperparamètres (`lr_config_t`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `learning_rate` | 0.01 | Taille du pas \(\eta\) |
| `max_iters` | 1000 | Nombre max d'itérations |
| `tolerance` | 1e-8 | Seuil de convergence sur \(\|\nabla L\|\) |

### Quand l'utiliser

| Avantage | Inconvénient |
|----------|--------------|
| Scalable à grands \(n\) | Convergence lente si \(\eta\) mal choisi |
| Pas besoin d'inverser une matrice | Solution approchée, pas exacte |
| Base pour SGD, mini-batch (futur) | Features non normalisées → convergence difficile |

## Métrique : R² (coefficient de détermination)

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- \(SS_{res} = \sum (y_i - \hat{y}_i)^2\) — erreur du modèle
- \(SS_{tot} = \sum (y_i - \bar{y})^2\) — variance totale

| R² | Interprétation |
|----|----------------|
| 1.0 | Ajustement parfait |
| 0.0 | Modèle aussi bon que prédire la moyenne |
| < 0 | Modèle pire que la moyenne |

## Exemple complet

```c
#include "yaj_ml/linear_regression.h"
#include <stdio.h>

int main(void)
{
    lr_model_t model;
    lr_config_t config;
    /* y = 2*x + 1 */
    const double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[] = {3.0, 5.0, 7.0, 9.0, 11.0};
    double pred;
    double r2;

    lr_init(&model, 1);
    lr_config_default(LR_METHOD_NORMAL_EQUATION, &config);
    lr_fit(&model, &config, X, y, 5, 1);

    lr_predict(&model, (double[]){6.0}, 1, &pred);
    lr_score(&model, X, y, 5, 1, &r2);

    printf("w = %.2f, b = %.2f\n", model.weights[0], model.weights[1]);
    printf("predict(6) = %.2f\n", pred);
    printf("R² = %.4f\n", r2);

    lr_free(&model);
    return 0;
}
```

Résultat attendu : `w = 2.00, b = 1.00, predict(6) = 13.00, R² = 1.0000`

## Utilitaire ajouté : `mat_solve`

La régression linéaire a nécessité l'ajout de `mat_solve()` dans la bibliothèque core.

Résout \(A\mathbf{x} = \mathbf{b}\) par **élimination de Gauss avec pivot partiel** :

1. Construire la matrice augmentée \([A | \mathbf{b}]\)
2. Pour chaque colonne : choisir le plus grand pivot (stabilité numérique)
3. Éliminer sous le pivot
4. Substitution arrière

Retourne `YAJ_ML_ERR_SINGULAR` si la matrice n'est pas inversible.

Voir [`src/matrix.c`](../../src/matrix.c) et [04_core_math.md](04_core_math.md).

## Tests

```bash
make test
```

Tests couverts :
- Équations normales 1D et 2D (poids exacts)
- Gradient descent (convergence vers les mêmes poids)
- Prédiction, R², erreurs (`NOT_FITTED`, `DIM`)

## Prochain modèle

**Perceptron** ou **Régression logistique** — les deux sont des modèles de classification binaire.
