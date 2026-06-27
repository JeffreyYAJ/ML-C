/**
 * @file linear_regression.c
 * @brief Ordinary least-squares linear regression.
 *
 * Algorithm reference: docs/fr/06_linear_regression.md
 */

#include "yaj_ml/linear_regression.h"

#include "yaj_ml/matrix.h"
#include "yaj_ml/vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static yaj_ml_status_t lr_validate_fit_args(const lr_model_t *model,
                                            const double *X, const double *y,
                                            size_t n_samples, size_t n_features)
{
    if (model == NULL || X == NULL || y == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (model->weights == NULL) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (n_samples == 0 || n_features == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    return YAJ_ML_OK;
}

static yaj_ml_status_t lr_fit_normal_equation(lr_model_t *model,
                                              const double *X, const double *y,
                                              size_t n_samples,
                                              size_t n_features)
{
    yaj_ml_status_t status;
    yaj_ml_mat_t X_mat = {0};
    yaj_ml_mat_t X_aug = {0};
    yaj_ml_mat_t Xt = {0};
    yaj_ml_mat_t XtX = {0};
    yaj_ml_vec_t y_vec = {0};
    yaj_ml_vec_t Xty = {0};
    yaj_ml_vec_t w_vec = {0};
    size_t n_weights;

    n_weights = n_features + 1U;

    status = mat_create_from_buffer(n_samples, n_features, X, &X_mat);
    if (status != YAJ_ML_OK) {
        return status;
    }

    status = mat_add_row(&X_mat, &X_aug);
    mat_free(&X_mat);
    if (status != YAJ_ML_OK) {
        return status;
    }

    status = mat_create(X_aug.cols, X_aug.rows, &Xt);
    if (status != YAJ_ML_OK) {
        mat_free(&X_aug);
        return status;
    }

    status = mat_transpose(&X_aug, &Xt);
    if (status != YAJ_ML_OK) {
        mat_free(&X_aug);
        mat_free(&Xt);
        return status;
    }

    status = mat_create(n_weights, n_weights, &XtX);
    if (status != YAJ_ML_OK) {
        mat_free(&X_aug);
        mat_free(&Xt);
        return status;
    }

    /* Normal equation: w = (X^T X)^(-1) X^T y  →  solve (X^T X) w = X^T y */
    status = mat_mul(&Xt, &X_aug, &XtX);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = vec_create(n_samples, &y_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }
    memcpy(y_vec.data, y, n_samples * sizeof(double));

    status = vec_create(n_weights, &Xty);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = mat_vec_mul(&Xt, &y_vec, &Xty);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = vec_create(n_weights, &w_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = mat_solve(&XtX, &Xty, &w_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    memcpy(model->weights, w_vec.data, n_weights * sizeof(double));
    model->fitted = true;
    model->method = LR_METHOD_NORMAL_EQUATION;

cleanup:
    mat_free(&X_aug);
    mat_free(&Xt);
    mat_free(&XtX);
    vec_free(&y_vec);
    vec_free(&Xty);
    vec_free(&w_vec);
    return status;
}

static yaj_ml_status_t lr_fit_gradient_descent(lr_model_t *model,
                                               const lr_config_t *config,
                                               const double *X, const double *y,
                                               size_t n_samples,
                                               size_t n_features)
{
    yaj_ml_status_t status;
    yaj_ml_mat_t X_mat = {0};
    yaj_ml_mat_t X_aug = {0};
    yaj_ml_mat_t Xt = {0};
    yaj_ml_vec_t w_vec = {0};
    yaj_ml_vec_t y_vec = {0};
    yaj_ml_vec_t y_hat = {0};
    yaj_ml_vec_t residual = {0};
    yaj_ml_vec_t gradient = {0};
    size_t n_weights;
    size_t iter;
    double grad_norm;
    double inv_n;
    double scale;

    n_weights = n_features + 1U;
    inv_n = 1.0 / (double)n_samples;

    status = mat_create_from_buffer(n_samples, n_features, X, &X_mat);
    if (status != YAJ_ML_OK) {
        return status;
    }

    status = mat_add_row(&X_mat, &X_aug);
    mat_free(&X_mat);
    if (status != YAJ_ML_OK) {
        return status;
    }

    status = mat_create(n_weights, n_samples, &Xt);
    if (status != YAJ_ML_OK) {
        mat_free(&X_aug);
        return status;
    }

    status = mat_transpose(&X_aug, &Xt);
    if (status != YAJ_ML_OK) {
        mat_free(&X_aug);
        mat_free(&Xt);
        return status;
    }

    status = vec_create(n_weights, &w_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }
    status = vec_set_zero(&w_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = vec_create(n_samples, &y_vec);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }
    memcpy(y_vec.data, y, n_samples * sizeof(double));

    status = vec_create(n_samples, &y_hat);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = vec_create(n_samples, &residual);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    status = vec_create(n_weights, &gradient);
    if (status != YAJ_ML_OK) {
        goto cleanup;
    }

    /*
     * Gradient descent on MSE: L = (1/n) ||Xw - y||^2
     * Gradient: (2/n) X^T (Xw - y)
     * Update:   w <- w - lr * gradient
     */
    for (iter = 0; iter < config->max_iters; ++iter) {
        size_t i;

        status = mat_vec_mul(&X_aug, &w_vec, &y_hat);
        if (status != YAJ_ML_OK) {
            goto cleanup;
        }

        for (i = 0; i < n_samples; ++i) {
            residual.data[i] = y_hat.data[i] - y_vec.data[i];
        }

        status = mat_vec_mul(&Xt, &residual, &gradient);
        if (status != YAJ_ML_OK) {
            goto cleanup;
        }

        scale = 2.0 * inv_n * config->learning_rate;
        status = vec_scale(scale, &gradient);
        if (status != YAJ_ML_OK) {
            goto cleanup;
        }

        for (i = 0; i < n_weights; ++i) {
            w_vec.data[i] -= gradient.data[i];
        }

        grad_norm = 0.0;
        for (i = 0; i < n_weights; ++i) {
            grad_norm += gradient.data[i] * gradient.data[i];
        }
        grad_norm = sqrt(grad_norm);

        if (grad_norm < config->tolerance) {
            break;
        }
    }

    memcpy(model->weights, w_vec.data, n_weights * sizeof(double));
    model->fitted = true;
    model->method = LR_METHOD_GRADIENT_DESCENT;
    status = YAJ_ML_OK;

cleanup:
    mat_free(&X_aug);
    mat_free(&Xt);
    vec_free(&w_vec);
    vec_free(&y_vec);
    vec_free(&y_hat);
    vec_free(&residual);
    vec_free(&gradient);
    return status;
}

yaj_ml_status_t lr_init(lr_model_t *model, size_t n_features)
{
    double *weights;

    if (model == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (n_features == 0) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    weights = calloc(n_features + 1U, sizeof(double));
    if (weights == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    model->weights = weights;
    model->n_features = n_features;
    model->fitted = false;
    model->method = LR_METHOD_NORMAL_EQUATION;
    return YAJ_ML_OK;
}

yaj_ml_status_t lr_config_default(lr_method_t method, lr_config_t *out)
{
    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    out->method = method;
    out->max_iters = 1000U;
    out->learning_rate = 0.01;
    out->tolerance = 1e-8;
    return YAJ_ML_OK;
}

yaj_ml_status_t lr_fit(lr_model_t *model, const lr_config_t *config,
                       const double *X, const double *y,
                       size_t n_samples, size_t n_features)
{
    yaj_ml_status_t status;
    lr_config_t default_config;

    status = lr_validate_fit_args(model, X, y, n_samples, n_features);
    if (status != YAJ_ML_OK) {
        return status;
    }

    if (config == NULL) {
        status = lr_config_default(LR_METHOD_NORMAL_EQUATION, &default_config);
        if (status != YAJ_ML_OK) {
            return status;
        }
        config = &default_config;
    }

    if (n_samples < n_features + 1U && config->method == LR_METHOD_NORMAL_EQUATION) {
        /* Underdetermined or exactly determined systems may be singular. */
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (config->method == LR_METHOD_NORMAL_EQUATION) {
        return lr_fit_normal_equation(model, X, y, n_samples, n_features);
    }

    return lr_fit_gradient_descent(model, config, X, y, n_samples, n_features);
}

yaj_ml_status_t lr_predict(const lr_model_t *model, const double *x,
                           size_t n_features, double *out)
{
    size_t i;
    double result;

    if (model == NULL || x == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    result = model->weights[n_features];
    for (i = 0; i < n_features; ++i) {
        result += model->weights[i] * x[i];
    }

    *out = result;
    return YAJ_ML_OK;
}

yaj_ml_status_t lr_score(const lr_model_t *model, const double *X,
                         const double *y, size_t n_samples,
                         size_t n_features, double *r2_out)
{
    yaj_ml_status_t status;
    size_t i;
    double y_mean;
    double ss_tot;
    double ss_res;
    double prediction;
    const double *row;

    if (model == NULL || X == NULL || y == NULL || r2_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_samples == 0 || n_features != model->n_features) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    y_mean = 0.0;
    for (i = 0; i < n_samples; ++i) {
        y_mean += y[i];
    }
    y_mean /= (double)n_samples;

    ss_tot = 0.0;
    ss_res = 0.0;

    for (i = 0; i < n_samples; ++i) {
        row = X + i * n_features;
        status = lr_predict(model, row, n_features, &prediction);
        if (status != YAJ_ML_OK) {
            return status;
        }

        ss_res += (y[i] - prediction) * (y[i] - prediction);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }

    if (ss_tot < 1e-15) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    *r2_out = 1.0 - (ss_res / ss_tot);
    return YAJ_ML_OK;
}

void lr_free(lr_model_t *model)
{
    if (model == NULL) {
        return;
    }

    free(model->weights);
    model->weights = NULL;
    model->n_features = 0;
    model->fitted = false;
}
