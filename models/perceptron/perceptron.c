/**
 * @file perceptron.c
 * @brief Rosenblatt perceptron for binary linear classification.
 *
 * Algorithm reference: docs/fr/09_perceptron.md
 */

#include "yaj_ml/perceptron.h"

#include <stdlib.h>

static double perc_dot_weights(const perc_model_t *model, const double *x,
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

static double perc_activate(double z)
{
    /* Step function: 1 if z > 0, else 0. */
    if (z > 0.0) {
        return 1.0;
    }
    return 0.0;
}

static yaj_ml_status_t perc_validate_labels(const double *y, size_t n_samples)
{
    size_t i;

    for (i = 0; i < n_samples; ++i) {
        if (y[i] != 0.0 && y[i] != 1.0) {
            return YAJ_ML_ERR_INVALID_ARG;
        }
    }

    return YAJ_ML_OK;
}

static yaj_ml_status_t perc_validate_fit_args(const perc_model_t *model,
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

    status = perc_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    return YAJ_ML_OK;
}

yaj_ml_status_t perc_init(perc_model_t *model, size_t n_features)
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
    model->converged = false;
    return YAJ_ML_OK;
}

yaj_ml_status_t perc_config_default(perc_config_t *out)
{
    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    out->max_epochs = 100U;
    out->learning_rate = 1.0;
    return YAJ_ML_OK;
}

yaj_ml_status_t perc_fit(perc_model_t *model, const perc_config_t *config,
                         const double *X, const double *y,
                         size_t n_samples, size_t n_features)
{
    yaj_ml_status_t status;
    perc_config_t default_config;
    size_t epoch;
    size_t i;
    size_t j;
    size_t errors;
    const double *row;
    double z;
    double prediction;
    double error;
    double lr;

    status = perc_validate_fit_args(model, X, y, n_samples, n_features);
    if (status != YAJ_ML_OK) {
        return status;
    }

    if (config == NULL) {
        status = perc_config_default(&default_config);
        if (status != YAJ_ML_OK) {
            return status;
        }
        config = &default_config;
    }

    lr = config->learning_rate;
    model->converged = false;

    /*
     * Rosenblatt perceptron learning rule (labels 0/1):
     *   prediction = 1 if w·x + b > 0 else 0
     *   on mistake: w <- w + lr * (y - prediction) * x_aug
     */
    for (epoch = 0; epoch < config->max_epochs; ++epoch) {
        errors = 0U;

        for (i = 0; i < n_samples; ++i) {
            row = X + i * n_features;
            z = perc_dot_weights(model, row, n_features);
            prediction = perc_activate(z);
            error = y[i] - prediction;

            if (error != 0.0) {
                ++errors;

                for (j = 0; j < n_features; ++j) {
                    model->weights[j] += lr * error * row[j];
                }
                model->weights[n_features] += lr * error;
            }
        }

        if (errors == 0U) {
            model->converged = true;
            break;
        }
    }

    model->fitted = true;
    return YAJ_ML_OK;
}

yaj_ml_status_t perc_predict(const perc_model_t *model, const double *x,
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

    z = perc_dot_weights(model, x, n_features);
    *class_out = perc_activate(z);
    return YAJ_ML_OK;
}

yaj_ml_status_t perc_score(const perc_model_t *model, const double *X,
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

    status = perc_validate_labels(y, n_samples);
    if (status != YAJ_ML_OK) {
        return status;
    }

    correct = 0U;
    for (i = 0; i < n_samples; ++i) {
        row = X + i * n_features;
        status = perc_predict(model, row, n_features, &prediction);
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

void perc_free(perc_model_t *model)
{
    if (model == NULL) {
        return;
    }

    free(model->weights);
    model->weights = NULL;
    model->n_features = 0U;
    model->fitted = false;
    model->converged = false;
}
