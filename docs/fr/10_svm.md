# SVM linéaire (Support Vector Machine)

Classificateur qui cherche la **frontière linéaire avec la plus grande marge** entre les deux classes.

## Idée clé

Parmi toutes les droites qui séparent les classes, le SVM choisit celle qui **maximise la distance** aux points les plus proches (vecteurs de support).

```
    classe 0  |  marge  |  classe 1
       o      |    |    |      o
          o   |    |    |   o
              |----|----|  ← frontière optimale (marge max)
```

## Modèle

Score linéaire : \(z = \mathbf{w} \cdot \mathbf{x} + b\)

Prédiction : classe 1 si \(z > 0\), sinon 0 (comme le perceptron).

La différence est dans **comment** \(\mathbf{w}\) et \(b\) sont appris.

## Fichiers

| Fichier | Rôle |
|---------|------|
| [`include/yaj_ml/svm.h`](../../include/yaj_ml/svm.h) | API publique |
| [`models/svm/svm.c`](../../models/svm/svm.c) | Implémentation |

## API

```c
svm_model_t model;
svm_config_t config;

svm_init(&model, n_features);
svm_config_default(&config);
svm_fit(&model, &config, X, y, n_samples, n_features);
svm_predict(&model, x, n_features, &class);
svm_score(&model, X, y, n_samples, n_features, &accuracy);
svm_free(&model);
```

Labels : **0.0** ou **1.0** (convertis en -1/+1 en interne).

## Fonction de coût (forme primale)

**Soft-margin SVM** avec hinge loss et régularisation L2 :

\[
L(\mathbf{w}, b) = \frac{\lambda}{2} \|\mathbf{w}\|^2 + \frac{1}{n} \sum_{i=1}^{n} \max(0,\ 1 - y_i \cdot z_i)
\]

- \(y_i \in \{-1, +1\}\) (converti depuis 0/1)
- \(z_i = \mathbf{w} \cdot \mathbf{x}_i + b\)
- \(\lambda\) = force de régularisation (`config.lambda`)

### Hinge loss

\[
\max(0,\ 1 - y \cdot z)
\]

| Situation | \(y \cdot z\) | Loss |
|-----------|---------------|------|
| Bien classé, large marge | > 1 | 0 |
| Bien classé, marge faible | entre 0 et 1 | > 0 |
| Mal classé | < 0 | > 1 |

Le modèle est pénalisé tant que la marge n'est pas au moins 1.

## Entraînement : gradient descent

YAJ-ML utilise la **descente de gradient par batch** sur la forme primale (pédagogique, sans solver dual ni SMO).

Pour chaque époque :

```
grad_w = lambda * w
grad_b = 0

pour chaque échantillon (x_i, y_i) :
    z = w·x + b
    si 1 - y_i * z > 0 :          # hinge actif
        grad_w -= y_i * x_i / n
        grad_b -= y_i / n

w -= lr * grad_w
b -= lr * grad_b
```

## Hyperparamètres

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `lambda` | 0.01 | Régularisation L2 (\(\uparrow\) → marge plus étroite, moins d'overfitting) |
| `learning_rate` | 0.01 | Pas de gradient descent |
| `max_iters` | 2000 | Époques max |
| `tolerance` | 1e-6 | Arrêt si \(\|\nabla L\|\) petit |

## Comparaison avec les autres classificateurs linéaires

| | Perceptron | Régression logistique | SVM |
|---|------------|----------------------|-----|
| Loss | Erreur 0/1 (perceptron) | Cross-entropy | Hinge |
| Marge | Non explicite | Non explicite | **Maximisée** |
| Régularisation | Non | Non (ici) | L2 sur \(\mathbf{w}\) |
| Probabilités | Non | Oui | Non |

## Exemple

```c
const double X[] = {0,0, 0,1, 1,0, 1,1};
const double y[] = {0, 0, 0, 1};

svm_init(&model, 2);
svm_config_default(&config);
config.learning_rate = 0.1;
config.lambda = 0.001;
svm_fit(&model, &config, X, y, 4, 2);
```

## Compiler l'exemple

```bash
make
gcc -std=c17 -Iinclude -o examples/svm_demo examples/svm.c \
    build-make/lib/libyaj_ml_svm.a \
    build-make/lib/libyaj_ml.a -lm
./examples/svm_demo
```

## Limites actuelles

- **Linéaire** seulement (pas de kernel trick / RBF)
- Forme **primale** avec gradient descent (pas de SMO dual)
- Classification **binaire**

Le kernel SVM (non-linéaire) pourrait être ajouté dans une version future via la forme duale.

## Bilan du projet

Avec le SVM, les **5 algorithmes** initiaux de YAJ-ML sont complets :

1. Régression linéaire
2. Régression logistique
3. Perceptron
4. KNN
5. SVM linéaire
