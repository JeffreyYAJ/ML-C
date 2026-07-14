/**
 * @file svm.c
 * @brief Linear SVM trained with primal hinge loss and batch gradient descent.
 *
 * Algorithm reference: docs/fr/10_svm.md
 */

#include "yaj_ml/svm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double svm_dot_weights(const svm_model_t *model, const double *x,
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

static double svm_label_to_pm(double y)
{
    /* Map 0/1 labels to -1/+1 for the SVM margin constraints. */
    return 2.0 * y - 1.0;
}

static double svm_activate(double z)
{
    if (z > 0.0) {
        return 1.0;
    }
    return 0.0;
}

static yaj_ml_status_t svm_validate_labels(const double *y, size_t n_samples)
{
    size_t i;

    for (i = 0; i < n_samples; ++i) {
        if (y[i] != 0.0 && y[i] != 1.0) {
            return YAJ_ML_ERR_INVALID_ARG;
        }
    }

    return YAJ_ML_OK;
}

static yaj_ml_status_t svm_validate_fit_args(const svm_model_t *model,
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

    if (n_samples == 0U || n_features == 0U) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    status = svm_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t svm_init(svm_model_t *model, size_t n_features)
{
    double *weights;

    if (model == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (n_features == 0U) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    weights = calloc(n_features + 1U, sizeof(double));
    if (weights == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    model->weights = weights;
    model->n_features = n_features;
    model->fitted = false;
    model->lambda = 0.0;
    return YAJ_ML_OK;
}

yaj_ml_status_t svm_config_default(svm_config_t *out)
{
    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    out->max_iters = 2000U;
    out->learning_rate = 0.01;
    out->lambda = 0.01;
    out->tolerance = 1e-6;
    return YAJ_ML_OK;
}

yaj_ml_status_t svm_fit(svm_model_t *model, const svm_config_t *config,
                        const double *X, const double *y,
                        size_t n_samples, size_t n_features)
{
    yaj_ml_status_t status;
    svm_config_t default_config;
    double *grad_w;
    size_t epoch;
    size_t i;
    size_t j;
    size_t n_weights;
    double inv_n;
    double grad_b;
    double grad_norm;
    double lr;
    double lambda;
    const double *row;
    double z;
    double y_pm;
    double margin;

    status = svm_validate_fit_args(model, X, y, n_samples, n_features);
    if (status != YAJ_ML_OK) {
        return status;
    }

    if (config == NULL) {
        status = svm_config_default(&default_config);
        if (status != YAJ_ML_OK) {
            return status;
        }
        config = &default_config;
    }

    n_weights = n_features + 1U;
    inv_n = 1.0 / (double)n_samples;
    lr = config->learning_rate;
    lambda = config->lambda;

    grad_w = calloc(n_weights, sizeof(double));
    if (grad_w == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    /*
     * Primal soft-margin SVM with hinge loss (labels in {-1,+1}):
     *   L = (lambda/2)||w||^2 + (1/n) sum max(0, 1 - y_i * (w·x_i + b))
     *
     * Batch subgradient descent each epoch.
     */
    for (epoch = 0; epoch < config->max_iters; ++epoch) {
        memset(grad_w, 0, n_weights * sizeof(double));
        grad_b = 0.0;

        for (i = 0; i < n_features; ++i) {
            grad_w[i] = lambda * model->weights[i];
        }

        for (i = 0; i < n_samples; ++i) {
            row = X + i * n_features;
            z = svm_dot_weights(model, row, n_features);
            y_pm = svm_label_to_pm(y[i]);
            margin = 1.0 - y_pm * z;

            if (margin > 0.0) {
                for (j = 0; j < n_features; ++j) {
                    grad_w[j] -= inv_n * y_pm * row[j];
                }
                grad_b -= inv_n * y_pm;
            }
        }

        for (i = 0; i < n_features; ++i) {
            model->weights[i] -= lr * grad_w[i];
        }
        model->weights[n_features] -= lr * grad_b;

        grad_norm = 0.0;
        for (i = 0; i < n_features; ++i) {
            grad_norm += grad_w[i] * grad_w[i];
        }
        grad_norm += grad_b * grad_b;
        grad_norm = sqrt(grad_norm);

        if (grad_norm < config->tolerance) {
            break;
        }
    }

    free(grad_w);
    model->fitted = true;
    model->lambda = lambda;
    return YAJ_ML_OK;
}

yaj_ml_status_t svm_predict(const svm_model_t *model, const double *x,
                            size_t n_features, double *class_out)
{
    double z;

    if (model == NULL || x == NULL || class_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    z = svm_dot_weights(model, x, n_features);
    *class_out = svm_activate(z);
    return YAJ_ML_OK;
}

yaj_ml_status_t svm_score(const svm_model_t *model, const double *X,
                          const double *y, size_t n_samples,
                          size_t n_features, double *accuracy_out)
{
    yaj_ml_status_t status;
    size_t i;
    size_t correct;
    const double *row;
    double prediction;

    if (model == NULL || X == NULL || y == NULL || accuracy_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_samples == 0U || n_features != model->n_features) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    status = svm_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    correct = 0U;
    for (i = 0; i < n_samples; ++i) {
        row = X + i * n_features;
        status = svm_predict(model, row, n_features, &prediction);
        if (status != YAJ_ML_OK) {
            return status;
        }

        if (prediction == y[i]) {
            ++correct;
        }
    }

    *accuracy_out = (double)correct / (double)n_samples;
    return YAJ_ML_OK;
}

void svm_free(svm_model_t *model)
{
    if (model == NULL) {
        return;
    }

    free(model->weights);
    model->weights = NULL;
    model->n_features = 0U;
    model->fitted = false;
    model->lambda = 0.0;
}
