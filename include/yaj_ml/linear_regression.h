#ifndef YAJ_ML_LINEAR_REGRESSION_H
#define YAJ_ML_LINEAR_REGRESSION_H

/**
 * @file linear_regression.h
 * @brief Ordinary least-squares linear regression.
 *
 * Fits y = w0*x0 + w1*x1 + ... + w_{p-1}*x_{p-1} + b
 * where b is stored as weights[n_features].
 *
 * Algorithm reference: docs/fr/06_linear_regression.md
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"

#include <stddef.h>
#include <stdbool.h>

/**
 * Training method for linear regression.
 */
typedef enum {
    LR_METHOD_NORMAL_EQUATION,  /**< Closed-form: w = (X^T X)^(-1) X^T y */
    LR_METHOD_GRADIENT_DESCENT, /**< Iterative gradient descent on MSE loss */
} lr_method_t;

/**
 * Hyperparameters for @ref lr_fit.
 */
typedef struct {
    lr_method_t method;
    size_t max_iters;     /**< Max iterations (gradient descent only). */
    double learning_rate; /**< Step size (gradient descent only). */
    double tolerance;     /**< Stop when ||gradient|| < tolerance. */
} lr_config_t;

/**
 * Fitted linear regression model.
 *
 * weights[0..n_features-1] are feature coefficients;
 * weights[n_features] is the bias (intercept).
 */
typedef struct {
    double *weights;
    size_t n_features;
    bool fitted;
    lr_method_t method;
} lr_model_t;

/**
 * @brief Initialize a linear regression model.
 *
 * @param model Output model. Must not be NULL.
 * @param n_features Number of input features (excluding bias).
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n_features)
 */
yaj_ml_status_t lr_init(lr_model_t *model, size_t n_features);

/**
 * @brief Fill @p config with sensible defaults for @p method.
 *
 * @param method Training method to configure.
 * @param out Output configuration. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t lr_config_default(lr_method_t method, lr_config_t *out);

/**
 * @brief Fit the model on training data.
 *
 * @param model Model to train. Must have been initialized with lr_init.
 * @param config Training hyperparameters. If NULL, defaults are used.
 * @param X Feature matrix in row-major order (n_samples * n_features). Must not be NULL.
 * @param y Target vector (n_samples). Must not be NULL.
 * @param n_samples Number of training examples. Must be > 0.
 * @param n_features Number of features per sample. Must match model->n_features.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity Normal equation: O(n*p^2 + p^3). Gradient descent: O(max_iters * n * p).
 */
yaj_ml_status_t lr_fit(lr_model_t *model, const lr_config_t *config,
                       const double *X, const double *y,
                       size_t n_samples, size_t n_features);

/**
 * @brief Predict a single target value.
 *
 * @param model Fitted model. Must not be NULL.
 * @param x Feature vector (n_features). Must not be NULL.
 * @param n_features Number of features. Must match model->n_features.
 * @param out Predicted value. Caller owns the storage.
 * @return YAJ_ML_OK on success, YAJ_ML_ERR_NOT_FITTED if not trained,
 *         or another error code on failure.
 *
 * @complexity O(n_features)
 */
yaj_ml_status_t lr_predict(const lr_model_t *model, const double *x,
                           size_t n_features, double *out);

/**
 * @brief Compute the coefficient of determination R² on a dataset.
 *
 * R² = 1 - SS_res / SS_tot, where SS_res = sum((y - y_hat)^2) and
 * SS_tot = sum((y - mean(y))^2).
 *
 * @param model Fitted model. Must not be NULL.
 * @param X Feature matrix in row-major order. Must not be NULL.
 * @param y Target vector. Must not be NULL.
 * @param n_samples Number of samples.
 * @param n_features Number of features per sample.
 * @param r2_out Output R² score. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n_samples * n_features)
 */
yaj_ml_status_t lr_score(const lr_model_t *model, const double *X,
                         const double *y, size_t n_samples,
                         size_t n_features, double *r2_out);

/**
 * @brief Release all memory owned by a model.
 *
 * @param model Model to destroy. Safe to call on a zero-initialized struct.
 */
void lr_free(lr_model_t *model);

#endif /* YAJ_ML_LINEAR_REGRESSION_H */
