#include "yaj_ml/vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

yaj_ml_status_t vec_create(size_t n, yaj_ml_vec_t *out)
{
    double *data;

    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (n == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    data = calloc(n, sizeof(double));
    if (data == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    out->n = n;
    out->data = data;
    return YAJ_ML_OK;
}

void vec_free(yaj_ml_vec_t *vec)
{
    if (vec == NULL) {
        return;
    }

    free(vec->data);
    vec->data = NULL;
    vec->n = 0;
}

yaj_ml_status_t vec_copy(const yaj_ml_vec_t *src, yaj_ml_vec_t *out)
{
    yaj_ml_status_t status;

    if (src == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (src->data == NULL || src->n == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = vec_create(src->n, out);
    if (status != YAJ_ML_OK) {
        return status;
    }

    memcpy(out->data, src->data, src->n * sizeof(double));
    return YAJ_ML_OK;
}

yaj_ml_status_t vec_dot(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b, double *out)
{
    size_t i;
    double sum;

    if (a == NULL || b == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || b->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (a->n != b->n) {
        return YAJ_ML_ERR_DIM;
    }

    sum = 0.0;
    for (i = 0; i < a->n; ++i) {
        sum += a->data[i] * b->data[i];
    }

    *out = sum;
    return YAJ_ML_OK;
}

yaj_ml_status_t vec_add(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b,
                        yaj_ml_vec_t *out)
{
    size_t i;

    if (a == NULL || b == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || b->data == NULL || out->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (a->n != b->n || a->n != out->n) {
        return YAJ_ML_ERR_DIM;
    }

    for (i = 0; i < a->n; ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t vec_sub(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b,
                        yaj_ml_vec_t *out)
{
    size_t i;

    if (a == NULL || b == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (a->data == NULL || b->data == NULL || out->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (a->n != b->n || a->n != out->n) {
        return YAJ_ML_ERR_DIM;
    }

    for (i = 0; i < a->n; ++i) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t vec_scale(double alpha, yaj_ml_vec_t *v)
{
    size_t i;

    if (v == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (v->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    for (i = 0; i < v->n; ++i) {
        v->data[i] *= alpha;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t vec_norm_l2(const yaj_ml_vec_t *v, double *out)
{
    size_t i;
    double sum_sq;

    if (v == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (v->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    /* Sum of squares is sufficient for L2 norm of moderate-sized vectors. */
    sum_sq = 0.0;
    for (i = 0; i < v->n; ++i) {
        sum_sq += v->data[i] * v->data[i];
    }

    *out = sqrt(sum_sq);
    return YAJ_ML_OK;
}

yaj_ml_status_t vec_fill(yaj_ml_vec_t *v, double value)
{
    size_t i;

    if (v == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (v->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    for (i = 0; i < v->n; ++i) {
        v->data[i] = value;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t vec_set_zero(yaj_ml_vec_t *v)
{
    if (v == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (v->data == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    memset(v->data, 0, v->n * sizeof(double));
    return YAJ_ML_OK;
}
