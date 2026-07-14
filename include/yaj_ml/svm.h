#ifndef YAJ_ML_SVM_H
#define YAJ_ML_SVM_H

/**
 * @file svm.h
 * @brief Linear Support Vector Machine for binary classification.
 *
 * Soft-margin linear SVM trained with batch gradient descent on the
 * primal hinge loss. Labels must be 0.0 or 1.0.
 *
 * Algorithm reference: docs/fr/10_svm.md
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"

#include <stddef.h>
#include <stdbool.h>

/**
 * Hyperparameters for @ref svm_fit.
 */
typedef struct {
    size_t max_iters;     /**< Maximum training epochs. */
    double learning_rate; /**< Gradient descent step size. */
    double lambda;        /**< L2 regularization strength on weights. */
    double tolerance;     /**< Stop when ||gradient|| < tolerance. */
} svm_config_t;

/**
 * Fitted linear SVM model.
 *
 * weights[0..n_features-1] are feature coefficients;
 * weights[n_features] is the bias (intercept).
 */
typedef struct {
    double *weights;
    size_t n_features;
    bool fitted;
    double lambda;
} svm_model_t;

/**
 * @brief Initialize a linear SVM model.
 *
 * @param model Output model. Must not be NULL.
 * @param n_features Number of input features (excluding bias).
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t svm_init(svm_model_t *model, size_t n_features);

/**
 * @brief Fill @p out with sensible default hyperparameters.
 *
 * @param out Output configuration. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t svm_config_default(svm_config_t *out);

/**
 * @brief Train the SVM with batch gradient descent on hinge loss.
 *
 * @param model Model to train.
 * @param config Hyperparameters. If NULL, defaults are used.
 * @param X Feature matrix in row-major order.
 * @param y Binary labels (0.0 or 1.0).
 * @param n_samples Number of training examples.
 * @param n_features Number of features per sample.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t svm_fit(svm_model_t *model, const svm_config_t *config,
                        const double *X, const double *y,
                        size_t n_samples, size_t n_features);

/**
 * @brief Predict class label (0 or 1) for a single sample.
 *
 * Class 1 if w·x + b > 0, else 0.
 *
 * @param model Fitted model.
 * @param x Feature vector (n_features).
 * @param n_features Number of features.
 * @param class_out Output class (0.0 or 1.0). Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t svm_predict(const svm_model_t *model, const double *x,
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
yaj_ml_status_t svm_score(const svm_model_t *model, const double *X,
                          const double *y, size_t n_samples,
                          size_t n_features, double *accuracy_out);

/**
 * @brief Release all memory owned by a model.
 *
 * @param model Model to destroy.
 */
void svm_free(svm_model_t *model);

#endif /* YAJ_ML_SVM_H */
