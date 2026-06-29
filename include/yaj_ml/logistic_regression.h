#ifndef YAJ_ML_LOGISTIC_REGRESSION_H
#define YAJ_ML_LOGISTIC_REGRESSION_H

/**
 * @file logistic_regression.h
 * @brief Binary logistic regression for classification.
 *
 * Models P(y=1|x) = sigmoid(w·x + b). Labels must be 0.0 or 1.0.
 *
 * Algorithm reference: docs/fr/07_logistic_regression.md
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"

#include <stddef.h>
#include <stdbool.h>

/**
 * Hyperparameters for @ref logreg_fit.
 */
typedef struct {
    size_t max_iters;     /**< Maximum gradient descent iterations. */
    double learning_rate; /**< Step size. */
    double tolerance;     /**< Stop when ||gradient|| < tolerance. */
    double threshold;     /**< Decision threshold for @ref logreg_predict. */
} logreg_config_t;

/**
 * Fitted binary logistic regression model.
 *
 * weights[0..n_features-1] are feature coefficients;
 * weights[n_features] is the bias (intercept).
 */
typedef struct {
    double *weights;
    size_t n_features;
    bool fitted;
    double threshold;
} logreg_model_t;

/**
 * @brief Initialize a logistic regression model.
 *
 * @param model Output model. Must not be NULL.
 * @param n_features Number of input features (excluding bias).
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_init(logreg_model_t *model, size_t n_features);

/**
 * @brief Fill @p out with sensible default hyperparameters.
 *
 * @param out Output configuration. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_config_default(logreg_config_t *out);

/**
 * @brief Fit the model with batch gradient descent on binary cross-entropy.
 *
 * @param model Model to train. Must have been initialized with logreg_init.
 * @param config Hyperparameters. If NULL, defaults are used.
 * @param X Feature matrix in row-major order (n_samples * n_features).
 * @param y Binary labels (0.0 or 1.0), length n_samples.
 * @param n_samples Number of training examples.
 * @param n_features Number of features per sample.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_fit(logreg_model_t *model, const logreg_config_t *config,
                           const double *X, const double *y,
                           size_t n_samples, size_t n_features);

/**
 * @brief Predict P(y=1|x) for a single sample.
 *
 * @param model Fitted model.
 * @param x Feature vector (n_features).
 * @param n_features Number of features.
 * @param proba_out Output probability in [0, 1]. Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_predict_proba(const logreg_model_t *model,
                                   const double *x, size_t n_features,
                                   double *proba_out);

/**
 * @brief Predict class label (0 or 1) using the model threshold.
 *
 * @param model Fitted model.
 * @param x Feature vector (n_features).
 * @param n_features Number of features.
 * @param class_out Output class (0.0 or 1.0). Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_predict(const logreg_model_t *model, const double *x,
                               size_t n_features, double *class_out);

/**
 * @brief Compute classification accuracy on a dataset.
 *
 * @param model Fitted model.
 * @param X Feature matrix in row-major order.
 * @param y True labels (0.0 or 1.0).
 * @param n_samples Number of samples.
 * @param n_features Number of features per sample.
 * @param accuracy_out Output accuracy in [0, 1]. Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t logreg_score(const logreg_model_t *model, const double *X,
                             const double *y, size_t n_samples,
                             size_t n_features, double *accuracy_out);

/**
 * @brief Release all memory owned by a model.
 *
 * @param model Model to destroy.
 */
void logreg_free(logreg_model_t *model);

#endif /* YAJ_ML_LOGISTIC_REGRESSION_H */
