#include "yaj_ml/matrix.h"

#include <stdlib.h>
#include <string.h>

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

    for (i = 0; i < src->rows; ++i) {
        for (j = 0; j < src->cols; ++j) {
            out->data[mat_index(i, out->cols, j)] =
                src->data[mat_index(i, src->cols, j)];
        }
        out->data[mat_index(i, out->cols, src->cols)] = 1.0;
    }

    return YAJ_ML_OK;
}
