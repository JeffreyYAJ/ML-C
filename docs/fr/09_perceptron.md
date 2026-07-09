# Perceptron

Classificateur linéaire binaire inventé par Frank Rosenblatt (1957). C'est le précurseur des réseaux de neurones modernes.

## Modèle

Score linéaire :

\[
z = w_0 x_0 + w_1 x_1 + \cdots + b
\]

Fonction d'activation **pas** (step function) :

\[
\hat{y} = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{sinon} \end{cases}
\]

Pas de sigmoïde — décision binaire directe.

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`include/yaj_ml/perceptron.h`](../../include/yaj_ml/perceptron.h) | API publique |
| [`models/perceptron/perceptron.c`](../../models/perceptron/perceptron.c) | Implémentation |

## API

```c
perc_model_t model;
perc_config_t config;

perc_init(&model, n_features);
perc_config_default(&config);
perc_fit(&model, &config, X, y, n_samples, n_features);
perc_predict(&model, x, n_features, &class);
perc_score(&model, X, y, n_samples, n_features, &accuracy);
perc_free(&model);
```

Labels : **0.0** ou **1.0** uniquement.

## Algorithme d'apprentissage

Pour chaque époque, parcourir tous les échantillons :

```
z = w·x + b
prediction = 1 si z > 0, sinon 0

si prediction != y :
    w <- w + lr * (y - prediction) * x
    b <- b + lr * (y - prediction)
```

Avec \(x_{aug} = [x_0, \ldots, x_{p-1}, 1]\), cela se réduit à :

\[
\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \cdot \mathbf{x}_{aug}
\]

### Convergence

- Si les données sont **linéairement séparables**, le perceptron converge en un nombre fini d'étapes (`model.converged = true`).
- Sinon (ex. XOR), il ne converge jamais — `converged` reste `false` après `max_epochs`.

C'est le **théorème de convergence du perceptron**.

## Hyperparamètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `learning_rate` | 1.0 | Pas \(\eta\) |
| `max_epochs` | 100 | Passes max sur le dataset |

## Comparaison avec la régression logistique

| | Perceptron | Régression logistique |
|---|------------|----------------------|
| Activation | Step (\(z > 0\)) | Sigmoïde |
| Sortie | 0 ou 1 | Probabilité \([0,1]\) |
| Entraînement | Mise à jour par erreur | Gradient sur BCE |
| Convergence | Garantie si séparable | Toujours converge (approx.) |
| Probabilités | Non | Oui |

## Exemple : porte AND

```c
const double X[] = {0,0, 0,1, 1,0, 1,1};
const double y[] = {0, 0, 0, 1};

perc_init(&model, 2);
perc_config_default(&config);
perc_fit(&model, &config, X, y, 4, 2);
/* model.converged == true, accuracy == 1.0 */
```

## XOR — limite du perceptron

| x0 | x1 | y |
|----|----|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

XOR n'est **pas linéairement séparable** → le perceptron ne peut pas atteindre 100 % d'accuracy. C'est historiquement ce qui a motivé le développement des réseaux multicouches (MLP).

## Compiler l'exemple

```bash
make
gcc -std=c17 -Iinclude -o examples/perc examples/perceptron.c \
    build-make/lib/libyaj_ml_perceptron.a \
    build-make/lib/libyaj_ml.a -lm
./examples/perc
```

## Champ `converged`

Après `perc_fit`, vérifie `model.converged` :

```c
if (model.converged) {
    printf("Solution trouvée en moins de %zu époques\n", config.max_epochs);
} else {
    printf("Pas de séparation linéaire parfaite\n");
}
```

## Prochain modèle

**SVM** (Support Vector Machine) — maximise la marge entre les classes au lieu de compter les erreurs.
