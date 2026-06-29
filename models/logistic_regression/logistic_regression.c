/**
 * @file logistic_regression.c
 * @brief Binary logistic regression trained with batch gradient descent.
 *
 * Algorithm reference: docs/fr/07_logistic_regression.md
 */

#include "yaj_ml/logistic_regression.h"

#include "yaj_ml/matrix.h"
#include "yaj_ml/vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double logreg_sigmoid(double z)
{
    /*
     * Stable sigmoid: for z >= 0 compute 1/(1+exp(-z)), else exp(z)/(1+exp(z)).
     * Avoids overflow when |z| is large.
     */
    if (z >= 0.0) {
        double ez = exp(-z);
        return 1.0 / (1.0 + ez);
    }

    {
        double ez = exp(z);
        return ez / (1.0 + ez);
    }
}

static double logreg_dot_weights(const logreg_model_t *model, const double *x,
                                   size_t n_features)
{
    size_t i;
    double z;

    z = model->weights[n_features];
    for (i = 0; i < n_features; ++i) {
        z += model->weights[i] * x[i];
    }

    return z;
}

static yaj_ml_status_t logreg_validate_labels(const double *y, size_t n_samples)
{
    size_t i;

    for (i = 0; i < n_samples; ++i) {
        if (y[i] != 0.0 && y[i] != 1.0) {
            return YAJ_ML_ERR_INVALID_ARG;
        }
    }

    return YAJ_ML_OK;
}

static yaj_ml_status_t logreg_validate_fit_args(const logreg_model_t *model,
                                                const double *X, const double *y,
                                                size_t n_samples,
                                                size_t n_features)
{
    yaj_ml_status_t status;

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

    status = logreg_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t logreg_init(logreg_model_t *model, size_t n_features)
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
    model->threshold = 0.5;
    return YAJ_ML_OK;
}

yaj_ml_status_t logreg_config_default(logreg_config_t *out)
{
    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    out->max_iters = 1000U;
    out->learning_rate = 0.1;
    out->tolerance = 1e-8;
    out->threshold = 0.5;
    return YAJ_ML_OK;
}

yaj_ml_status_t logreg_fit(logreg_model_t *model, const logreg_config_t *config,
                           const double *X, const double *y,
                           size_t n_samples, size_t n_features)
{
    yaj_ml_status_t status;
    logreg_config_t default_config;
    yaj_ml_mat_t X_mat = {0};
    yaj_ml_mat_t X_aug = {0};
    yaj_ml_mat_t Xt = {0};
    yaj_ml_vec_t w_vec = {0};
    yaj_ml_vec_t y_vec = {0};
    yaj_ml_vec_t probs = {0};
    yaj_ml_vec_t residual = {0};
    yaj_ml_vec_t gradient = {0};
    size_t n_weights;
    size_t iter;
    double inv_n;
    double grad_norm;
    double scale;

    status = logreg_validate_fit_args(model, X, y, n_samples, n_features);
    if (status != YAJ_ML_OK) {
        return status;
    }

    if (config == NULL) {
        status = logreg_config_default(&default_config);
        if (status != YAJ_ML_OK) {
            return status;
        }
        config = &default_config;
    }

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

    status = vec_create(n_samples, &probs);
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
     * Batch gradient descent on binary cross-entropy:
     *   gradient = (1/n) X^T (sigmoid(Xw) - y)
     *   w <- w - lr * gradient
     */
    for (iter = 0; iter < config->max_iters; ++iter) {
        size_t i;

        status = mat_vec_mul(&X_aug, &w_vec, &probs);
        if (status != YAJ_ML_OK) {
            goto cleanup;
        }

        for (i = 0; i < n_samples; ++i) {
            probs.data[i] = logreg_sigmoid(probs.data[i]);
            residual.data[i] = probs.data[i] - y_vec.data[i];
        }

        status = mat_vec_mul(&Xt, &residual, &gradient);
        if (status != YAJ_ML_OK) {
            goto cleanup;
        }

        scale = inv_n * config->learning_rate;
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
    model->threshold = config->threshold;
    status = YAJ_ML_OK;

cleanup:
    mat_free(&X_aug);
    mat_free(&Xt);
    vec_free(&w_vec);
    vec_free(&y_vec);
    vec_free(&probs);
    vec_free(&residual);
    vec_free(&gradient);
    return status;
}

yaj_ml_status_t logreg_predict_proba(const logreg_model_t *model, const double *x,
                                   size_t n_features, double *proba_out)
{
    double z;

    if (model == NULL || x == NULL || proba_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    z = logreg_dot_weights(model, x, n_features);
    *proba_out = logreg_sigmoid(z);
    return YAJ_ML_OK;
}

yaj_ml_status_t logreg_predict(const logreg_model_t *model, const double *x,
                               size_t n_features, double *class_out)
{
    yaj_ml_status_t status;
    double proba;

    status = logreg_predict_proba(model, x, n_features, &proba);
    if (status != YAJ_ML_OK) {
        return status;
    }

    if (proba >= model->threshold) {
        *class_out = 1.0;
    } else {
        *class_out = 0.0;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t logreg_score(const logreg_model_t *model, const double *X,
                             const double *y, size_t n_samples,
                             size_t n_features, double *accuracy_out)
{
    yaj_ml_status_t status;
    size_t i;
    size_t correct;
    const double *row;
    double predicted;
    double proba;

    if (model == NULL || X == NULL || y == NULL || accuracy_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_samples == 0 || n_features != model->n_features) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = logreg_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    correct = 0U;
    for (i = 0; i < n_samples; ++i) {
        row = X + i * n_features;
        status = logreg_predict_proba(model, row, n_features, &proba);
        if (status != YAJ_ML_OK) {
            return status;
        }

        predicted = (proba >= model->threshold) ? 1.0 : 0.0;
        if (predicted == y[i]) {
            ++correct;
        }
    }

    *accuracy_out = (double)correct / (double)n_samples;
    return YAJ_ML_OK;
}

void logreg_free(logreg_model_t *model)
{
    if (model == NULL) {
        return;
    }

    free(model->weights);
    model->weights = NULL;
    model->n_features = 0;
    model->fitted = false;
    model->threshold = 0.5;
}
