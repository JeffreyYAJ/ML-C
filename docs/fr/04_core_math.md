# Mathématiques de base : vecteurs et matrices

Ce document explique les **algorithmes** implémentés dans `src/vector.c` et `src/matrix.c`, avec les formules mathématiques et le lien vers le code source.

## Notations

| Symbole | Signification |
|---------|---------------|
| \(\mathbf{a}, \mathbf{b}\) | Vecteurs (tableaux de `double`) |
| \(A, B\) | Matrices |
| \(n\) | Dimension d'un vecteur |
| \(A_{ij}\) | Élément ligne \(i\), colonne \(j\) de \(A\) |

En code, un vecteur est `yaj_ml_vec_t` et une matrice est `yaj_ml_mat_t`.

---

## Vecteurs

### Structure et stockage

```c
typedef struct {
    size_t n;       // longueur
    double *data;   // tableau contigu : data[0], data[1], ..., data[n-1]
} yaj_ml_vec_t;
```

Les éléments sont **contigus en mémoire** (cache-friendly). L'index `i` accède directement à `data[i]`.

**Fichiers** : [`include/yaj_ml/vector.h`](../../include/yaj_ml/vector.h), [`src/vector.c`](../../src/vector.c)

### Création (`vec_create`)

**Algorithme** :
1. Valider `n > 0` et `out != NULL`
2. Allouer `n * sizeof(double)` avec `calloc` (initialise à zéro)
3. Stocker `n` et le pointeur dans `out`

**Complexité** : O(n) pour l'initialisation à zéro.

### Produit scalaire (`vec_dot`)

**Formule** :

\[
\mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} a_i \cdot b_i
\]

**Exemple** : \(\mathbf{a} = [1, 2, 3]\), \(\mathbf{b} = [4, 5, 6]\)

\[
1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32
\]

**Implémentation** (boucle simple) :

```c
sum = 0.0;
for (i = 0; i < a->n; ++i) {
    sum += a->data[i] * b->data[i];
}
```

**Usage ML** : produit scalaire entre features et poids (régression, perceptron, SVM).

**Complexité** : O(n)

### Addition et soustraction (`vec_add`, `vec_sub`)

**Formule** :

\[
\text{out}_i = a_i \pm b_i \quad \forall i
\]

Opération **élément par élément**. Les trois vecteurs doivent avoir la même longueur.

**Usage ML** : calcul de gradients (\(\nabla = \nabla_{old} - \eta \cdot \text{grad}\)).

### Multiplication par un scalaire (`vec_scale`)

**Formule** :

\[
\mathbf{v} \leftarrow \alpha \cdot \mathbf{v}
\]

Modifie le vecteur **en place** : chaque élément est multiplié par \(\alpha\).

**Usage ML** : mise à l'échelle du learning rate, normalisation partielle.

### Norme L2 (`vec_norm_l2`)

**Formule** :

\[
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=0}^{n-1} v_i^2}
\]

**Exemple** : \(\mathbf{v} = [3, 4]\) → \(\|\mathbf{v}\|_2 = \sqrt{9 + 16} = 5\)

**Implémentation** :

```c
sum_sq = 0.0;
for (i = 0; i < v->n; ++i) {
    sum_sq += v->data[i] * v->data[i];
}
*out = sqrt(sum_sq);
```

**Note stabilité** : on accumule d'abord les carrés, puis on prend la racine une seule fois. Pour des vecteurs très grands, des variantes type Kahan summation pourraient être ajoutées plus tard.

**Usage ML** : normalisation, régularisation L2, calcul de distances euclidiennes (KNN).

---

## Matrices

### Structure et stockage row-major

```c
typedef struct {
    size_t rows;
    size_t cols;
    double *data;   // row-major : (i,j) → data[i * cols + j]
} yaj_ml_mat_t;
```

**Row-major** signifie que les lignes sont stockées bout à bout :

```
Matrice 2×3 :          data[]:
[ a00  a01  a02 ]  →  [ a00, a01, a02, a10, a11, a12 ]
[ a10  a11  a12 ]
```

**Pourquoi row-major ?** C'est la convention naturelle en C. Quand on parcourt `for i, for j`, on accède à la mémoire séquentiellement → bon pour le cache CPU.

**Fichiers** : [`include/yaj_ml/matrix.h`](../../include/yaj_ml/matrix.h), [`src/matrix.c`](../../src/matrix.c)

### Accès aux éléments (`mat_get`, `mat_set`)

**Formule d'index** :

\[
\text{index}(i, j) = i \times \text{cols} + j
\]

Fonction interne `mat_index(row, cols, col)` encapsule ce calcul.

### Multiplication matricielle (`mat_mul`)

**Formule** : si \(A\) est \(m \times k\) et \(B\) est \(k \times n\), alors \(C = AB\) est \(m \times n\) :

\[
C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{kj}
\]

**Exemple** :

\[
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\times
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\]

Car \(C_{00} = 1 \cdot 5 + 2 \cdot 7 = 19\), etc.

**Triple boucle naïve** :

```c
for (i = 0; i < a->rows; ++i) {
    for (j = 0; j < b->cols; ++j) {
        sum = 0.0;
        for (k = 0; k < a->cols; ++k) {
            sum += A[i,k] * B[k,j];
        }
        C[i,j] = sum;
    }
}
```

**Complexité** : O(m × n × k) — naïf mais correct. Des optimisations (blocking, SIMD) viendront plus tard si nécessaire.

**Usage ML** : régression linéaire (\(\hat{y} = Xw\)), propagation dans les réseaux simples.

### Transposition (`mat_transpose`)

**Formule** :

\[
(A^T)_{ij} = A_{ji}
\]

Une matrice \(m \times n\) devient \(n \times m\).

**Exemple** :

\[
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T
=
\begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
\]

**Complexité** : O(m × n)

**Usage ML** : équations normales (\(X^T X\)), calcul de gradients.

### Multiplication matrice-vecteur (`mat_vec_mul`)

**Formule** : si \(A\) est \(m \times n\) et \(\mathbf{v}\) est de dimension \(n\) :

\[
(\mathbf{y})_i = \sum_{j=0}^{n-1} A_{ij} \cdot v_j
\]

C'est une **combinaison linéaire** de chaque ligne de \(A\) avec les coefficients de \(\mathbf{v}\).

**Exemple** :

\[
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 \\ 6 \end{bmatrix}
=
\begin{bmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{bmatrix}
=
\begin{bmatrix} 17 \\ 39 \end{bmatrix}
\]

**Usage ML** : prédiction \(\hat{y} = X\mathbf{w} + b\) (sans terme de biais explicite si on utilise `mat_add_row`).

**Complexité** : O(m × n)

### Ajout de colonne de biais (`mat_add_row`)

**Problème ML** : la régression linéaire calcule \(\hat{y} = X\mathbf{w} + b\). On veut absorber le biais \(b\) dans un seul produit matriciel.

**Astuce** : ajouter une colonne de **1** à la matrice \(X\) et inclure \(b\) dans le vecteur de poids.

**Transformation** :

\[
\begin{bmatrix} x_1 & x_2 \end{bmatrix}
\quad \rightarrow \quad
\begin{bmatrix} x_1 & x_2 & 1 \end{bmatrix}
\]

Matrice \(m \times n\) → matrice \(m \times (n+1)\), dernière colonne = `1.0`.

**Formule** :

\[
\text{out}_{i,j} = \begin{cases} \text{src}_{i,j} & \text{si } j < n \\ 1.0 & \text{si } j = n \end{cases}
\]

Ensuite : \(\hat{y} = X_{aug} \cdot [\mathbf{w}; b]\) en un seul `mat_vec_mul`.

**Usage ML** : régression linéaire, régression logistique.

---

## Récapitulatif des complexités

| Fonction | Complexité | Usage ML principal |
|----------|------------|-------------------|
| `vec_dot` | O(n) | Score linéaire, similarité |
| `vec_add/sub` | O(n) | Mise à jour de poids |
| `vec_norm_l2` | O(n) | Distance euclidienne (KNN) |
| `mat_mul` | O(mnk) | Équations normales |
| `mat_transpose` | O(mn) | \(X^T X\) |
| `mat_vec_mul` | O(mn) | Prédiction \(\hat{y} = Xw\) |
| `mat_add_row` | O(mn) | Terme de biais |

---

## Lien avec les futurs modèles

| Modèle | Fonctions core utilisées |
|--------|-------------------------|
| Linear Regression | `mat_add_row`, `mat_mul`, `mat_transpose`, `mat_vec_mul` |
| Logistic Regression | idem + `vec_add`, `vec_scale` pour gradient descent |
| Perceptron | `vec_dot`, `vec_add`, `vec_scale` |
| KNN | `vec_norm_l2` (distance), `vec_sub` |
| SVM | `vec_dot`, `vec_norm_l2`, `mat_vec_mul` |

Chaque modèle sera documenté de la même façon quand il sera implémenté.
