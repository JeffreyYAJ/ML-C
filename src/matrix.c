/**
 * @file matrix.c
 * @brief Dense matrix operations for YAJ-ML.
 *
 * Matrices are stored in row-major order: element (row, col) lives at
 * data[row * cols + col]. See docs/fr/04_core_math.md for formulas.
 */

#include "yaj_ml/matrix.h"

#include <stdlib.h>
#include <string.h>

/* Row-major linear index: A[row,col] -> data[row * cols + col]. */
static size_t mat_index(size_t row, size_t cols, size_t col)
{
    return row * cols + col;
}

yaj_ml_status_t mat_create(size_t rows, size_t cols, yaj_ml_mat_t *out)
{
    double *data;

    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (rows == 0 || cols == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    data = calloc(rows * cols, sizeof(double));
    if (data == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    out->rows = rows;
    out->cols = cols;
    out->data = data;
    return YAJ_ML_OK;
}

void mat_free(yaj_ml_mat_t *mat)
{
    if (mat == NULL) {
        return;
    }

    free(mat->data);
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

yaj_ml_status_t mat_copy(const yaj_ml_mat_t *src, yaj_ml_mat_t *out)
{
    yaj_ml_status_t status;
    size_t count;

    if (src == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (src->data == NULL || src->rows == 0 || src->cols == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = mat_create(src->rows, src->cols, out);
    if (status != YAJ_ML_OK) {
        return status;
    }

    count = src->rows * src->cols;
    memcpy(out->data, src->data, count * sizeof(double));
    return YAJ_ML_OK;
}

yaj_ml_status_t mat_create_from_buffer(size_t rows, size_t cols,
                                      const double *data, yaj_ml_mat_t *out)
{
    yaj_ml_status_t status;
    size_t count;

    if (data == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (rows == 0 || cols == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = mat_create(rows, cols, out);
    if (status != YAJ_ML_OK) {
        return status;
    }

    count = rows * cols;
    memcpy(out->data, data, count * sizeof(double));
    return YAJ_ML_OK;
}

yaj_ml_status_t mat_get(const yaj_ml_mat_t *mat, size_t row, size_t col,
                        double *out)
{
    if (mat == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (mat->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (row >= mat->rows || col >= mat->cols) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    *out = mat->data[mat_index(row, mat->cols, col)];
    return YAJ_ML_OK;
}

yaj_ml_status_t mat_set(yaj_ml_mat_t *mat, size_t row, size_t col, double value)
{
    if (mat == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (mat->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (row >= mat->rows || col >= mat->cols) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    mat->data[mat_index(row, mat->cols, col)] = value;
    return YAJ_ML_OK;
}

yaj_ml_status_t mat_mul(const yaj_ml_mat_t *a, const yaj_ml_mat_t *b,
                        yaj_ml_mat_t *out)
{
    size_t i;
    size_t j;
    size_t k;
    double sum;

    if (a == NULL || b == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || b->data == NULL || out->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (a->cols != b->rows) {
        return YAJ_ML_ERR_DIM;
    }

    if (out->rows != a->rows || out->cols != b->cols) {
        return YAJ_ML_ERR_DIM;
    }

    /*
     * Naive matrix multiplication: C = A * B.
     * C[i,j] = sum_k A[i,k] * B[k,j].
     * Complexity O(m*n*k). Correct first; blocked/SIMD variants may follow.
     */
    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < b->cols; ++j) {
            sum = 0.0;
            for (k = 0; k < a->cols; ++k) {
                sum += a->data[mat_index(i, a->cols, k)] *
                       b->data[mat_index(k, b->cols, j)];
            }
            out->data[mat_index(i, out->cols, j)] = sum;
        }
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t mat_transpose(const yaj_ml_mat_t *a, yaj_ml_mat_t *out)
{
    size_t i;
    size_t j;

    if (a == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || out->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (out->rows != a->cols || out->cols != a->rows) {
        return YAJ_ML_ERR_DIM;
    }

    /* Transpose: out[j,i] = a[i,j]. Swaps rows and columns. */
    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            out->data[mat_index(j, out->cols, i)] =
                a->data[mat_index(i, a->cols, j)];
        }
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t mat_vec_mul(const yaj_ml_mat_t *mat, const yaj_ml_vec_t *vec,
                            yaj_ml_vec_t *out)
{
    size_t i;
    size_t j;
    double sum;

    if (mat == NULL || vec == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (mat->data == NULL || vec->data == NULL || out->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (mat->cols != vec->n) {
        return YAJ_ML_ERR_DIM;
    }

    if (out->n != mat->rows) {
        return YAJ_ML_ERR_DIM;
    }

    /*
     * Matrix-vector product: y = A * x.
     * y[i] = sum_j A[i,j] * x[j] — dot product of row i with x.
     * Core operation for prediction y_hat = X * w in linear models.
     */
    for (i = 0; i < mat->rows; ++i) {
        sum = 0.0;
        for (j = 0; j < mat->cols; ++j) {
            sum += mat->data[mat_index(i, mat->cols, j)] * vec->data[j];
        }
        out->data[i] = sum;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t mat_add_row(const yaj_ml_mat_t *src, yaj_ml_mat_t *out)
{
    size_t i;
    size_t j;
    yaj_ml_status_t status;

    if (src == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (src->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (out->data != NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = mat_create(src->rows, src->cols + 1, out);
    if (status != YAJ_ML_OK) {
        return status;
    }

    /*
     * Augment X with a bias column of ones so that y = X_aug * [w; b]
     * replaces y = X*w + b with a single matrix-vector multiply.
     * Output shape: (m x (n+1)) where the last column is always 1.0.
     */
    for (i = 0; i < src->rows; ++i) {
        for (j = 0; j < src->cols; ++j) {
            out->data[mat_index(i, out->cols, j)] =
                src->data[mat_index(i, src->cols, j)];
        }
        out->data[mat_index(i, out->cols, src->cols)] = 1.0;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t mat_solve(const yaj_ml_mat_t *a, const yaj_ml_vec_t *b,
                          yaj_ml_vec_t *x)
{
    size_t n;
    size_t i;
    size_t j;
    size_t k;
    size_t pivot_row;
    double pivot_val;
    double max_val;
    double factor;
    double *aug;
    double abs_val;

    if (a == NULL || b == NULL || x == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || b->data == NULL || x->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (a->rows != a->cols) {
        return YAJ_ML_ERR_DIM;
    }

    n = a->rows;

    if (b->n != n || x->n != n) {
        return YAJ_ML_ERR_DIM;
    }

    /*
     * Build augmented matrix [A | b] as a flat row-major array (n x (n+1)).
     * Gaussian elimination with partial pivoting finds x such that A*x = b.
     */
    aug = malloc(n * (n + 1U) * sizeof(double));
    if (aug == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            aug[i * (n + 1U) + j] = a->data[mat_index(i, n, j)];
        }
        aug[i * (n + 1U) + n] = b->data[i];
    }

    for (k = 0; k < n; ++k) {
        /* Partial pivoting: pick the largest pivot in column k. */
        pivot_row = k;
        max_val = 0.0;
        for (i = k; i < n; ++i) {
            abs_val = aug[i * (n + 1U) + k];
            if (abs_val < 0.0) {
                abs_val = -abs_val;
            }
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = i;
            }
        }

        if (max_val < 1e-12) {
            free(aug);
            return YAJ_ML_ERR_SINGULAR;
        }

        if (pivot_row != k) {
            for (j = k; j <= n; ++j) {
                double tmp = aug[k * (n + 1U) + j];
                aug[k * (n + 1U) + j] = aug[pivot_row * (n + 1U) + j];
                aug[pivot_row * (n + 1U) + j] = tmp;
            }
        }

        pivot_val = aug[k * (n + 1U) + k];

        /* Eliminate entries below the pivot. */
        for (i = k + 1U; i < n; ++i) {
            factor = aug[i * (n + 1U) + k] / pivot_val;
            aug[i * (n + 1U) + k] = 0.0;
            for (j = k + 1U; j <= n; ++j) {
                aug[i * (n + 1U) + j] -= factor * aug[k * (n + 1U) + j];
            }
        }
    }

    /* Back substitution. */
    for (i = n; i-- > 0U;) {
        double sum = aug[i * (n + 1U) + n];
        for (j = i + 1U; j < n; ++j) {
            sum -= aug[i * (n + 1U) + j] * x->data[j];
        }
        x->data[i] = sum / aug[i * (n + 1U) + i];
    }

    free(aug);
    return YAJ_ML_OK;
}
