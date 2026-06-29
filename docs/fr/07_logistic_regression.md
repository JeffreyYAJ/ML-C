# Régression logistique

Modèle de **classification binaire** : prédit la probabilité qu'un échantillon appartienne à la classe 1.

## Modèle mathématique

\[
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b)
\]

où \(\sigma\) est la **fonction sigmoïde** :

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- Sortie dans \([0, 1]\) → interprétable comme probabilité
- Labels d'entraînement : **0.0** ou **1.0** uniquement

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`include/yaj_ml/logistic_regression.h`](../../include/yaj_ml/logistic_regression.h) | API publique |
| [`models/logistic_regression/logistic_regression.c`](../../models/logistic_regression/logistic_regression.c) | Implémentation |

## API

```c
logreg_model_t model;
logreg_config_t config;

logreg_init(&model, n_features);
logreg_config_default(&config);
logreg_fit(&model, &config, X, y, n_samples, n_features);
logreg_predict_proba(&model, x, n_features, &proba);  /* P(y=1|x) */
logreg_predict(&model, x, n_features, &class);        /* 0 ou 1 */
logreg_score(&model, X, y, n_samples, n_features, &accuracy);
logreg_free(&model);
```

### Format des données

Identique à la régression linéaire :
- `X` : row-major, `X[i * n_features + j]`
- `y` : labels **0.0** ou **1.0** (pas 0/1 entiers obligatoires, mais doubles exacts)

## Fonction sigmoïde (implémentation)

La sigmoïde mappe un score linéaire \(z = \mathbf{w}\cdot\mathbf{x} + b\) vers une probabilité :

| \(z\) | \(\sigma(z)\) |
|-------|---------------|
| \(-\infty\) | 0 |
| 0 | 0.5 |
| \(+\infty\) | 1 |

**Stabilité numérique** dans `logreg_sigmoid()` :

```c
if (z >= 0)
    return 1 / (1 + exp(-z));
else
    return exp(z) / (1 + exp(z));
```

Évite `exp(+grand nombre)` qui provoque un overflow.

## Entraînement : gradient descent sur BCE

Pas de solution analytique fermée (contrairement à la régression linéaire). On minimise la **binary cross-entropy** :

\[
L(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]
\]

où \(p_i = \sigma(\mathbf{w} \cdot \mathbf{x}_i)\).

### Gradient

\[
\nabla L = \frac{1}{n} X^T (\mathbf{p} - \mathbf{y})
\]

### Mise à jour

\[
\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla L
\]

### Hyperparamètres (`logreg_config_t`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `learning_rate` | 0.1 | Pas \(\eta\) |
| `max_iters` | 1000 | Itérations max |
| `tolerance` | 1e-8 | Arrêt si \(\|\nabla L\| < \text{tolerance}\) |
| `threshold` | 0.5 | Seuil pour `logreg_predict` |

## Prédiction

```c
double proba;
logreg_predict_proba(&model, x, n_features, &proba);
/* proba ∈ [0, 1] */

double cls;
logreg_predict(&model, x, n_features, &cls);
/* cls = 1.0 si proba >= threshold, sinon 0.0 */
```

## Métrique : accuracy

\[
\text{accuracy} = \frac{\text{nombre de prédictions correctes}}{n}
\]

Retournée par `logreg_score()`.

## Exemple : porte AND

| x0 | x1 | y |
|----|----|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

```c
#include "yaj_ml/logistic_regression.h"
#include <stdio.h>

int main(void)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    const double X[] = {0,0, 0,1, 1,0, 1,1};
    const double y[] = {0, 0, 0, 1};
    double proba;
    double cls;

    logreg_init(&model, 2);
    logreg_config_default(&config);
    config.learning_rate = 0.5;
    config.max_iters = 10000;

    logreg_fit(&model, &config, X, y, 4, 2);

    logreg_predict_proba(&model, (double[]){1, 1}, 2, &proba);
    logreg_predict(&model, (double[]){1, 1}, 2, &cls);

    printf("AND(1,1): proba=%.3f, classe=%.0f\n", proba, cls);

    logreg_free(&model);
    return 0;
}
```

## Comparaison avec la régression linéaire

| | Régression linéaire | Régression logistique |
|---|---------------------|----------------------|
| Type | Régression (valeur continue) | Classification binaire |
| Sortie | \(\hat{y} \in \mathbb{R}\) | \(P(y=1) \in [0,1]\) |
| Labels | N'importe quel `double` | 0.0 ou 1.0 |
| Fonction d'activation | Aucune | Sigmoïde |
| Loss | MSE | Binary cross-entropy |
| Entraînement | Équations normales ou GD | GD uniquement |
| Score | R² | Accuracy |

## Compiler l'exemple

```bash
cd ~/Desktop/ML-C
make

gcc -std=c17 -Iinclude -o examples/logreg examples/logistic_regression.c \
    build-make/lib/libyaj_ml_logistic_regression.a \
    build-make/lib/libyaj_ml.a -lm

./examples/logreg
```

## Tests

```bash
make test
```

Tests couverts : séparation 1D, porte AND, labels invalides, erreurs API.

## Limites actuelles

- Classification **binaire** seulement (pas multi-classe)
- Batch gradient descent (pas de mini-batch / SGD)
- Pas de régularisation L2 (à ajouter plus tard)

## Prochain modèle

**Perceptron** — classification par mise à jour de poids basée sur des erreurs, sans sigmoïde.
