#ifndef YAJ_ML_KNN_H
#define YAJ_ML_KNN_H

/**
 * @file knn.h
 * @brief k-Nearest Neighbors for classification and regression.
 *
 * Instance-based learner: fit stores the training set; predict queries the
 * k closest training points by Euclidean distance.
 *
 * Algorithm reference: docs/fr/08_knn.md
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"

#include <stddef.h>
#include <stdbool.h>

/**
 * Prediction task for KNN.
 */
typedef enum {
    KNN_TASK_CLASSIFICATION, /**< Majority vote among neighbor labels. */
    KNN_TASK_REGRESSION,     /**< Mean of neighbor target values. */
} knn_task_t;

/**
 * Hyperparameters for @ref knn_fit.
 */
typedef struct {
    size_t k;         /**< Number of neighbors (must be >= 1 and <= n_samples). */
    knn_task_t task;  /**< Classification or regression. */
} knn_config_t;

/**
 * k-Nearest Neighbors model.
 *
 * Stores a copy of the training data after @ref knn_fit.
 */
typedef struct {
    double *X;
    double *y;
    size_t n_samples;
    size_t n_features;
    size_t k;
    knn_task_t task;
    bool fitted;
} knn_model_t;

/**
 * @brief Initialize an empty KNN model.
 *
 * @param model Output model. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t knn_init(knn_model_t *model);

/**
 * @brief Fill @p out with sensible default hyperparameters.
 *
 * Defaults: k=3, classification task.
 *
 * @param out Output configuration. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t knn_config_default(knn_config_t *out);

/**
 * @brief Store a copy of the training data (lazy learning).
 *
 * @param model Model to populate. Must have been initialized with knn_init.
 * @param config Hyperparameters. If NULL, defaults are used.
 * @param X Feature matrix in row-major order (n_samples * n_features).
 * @param y Labels (classification) or targets (regression), length n_samples.
 * @param n_samples Number of training examples. Must be > 0.
 * @param n_features Number of features per sample. Must be > 0.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n_samples * n_features) for the data copy.
 */
yaj_ml_status_t knn_fit(knn_model_t *model, const knn_config_t *config,
                        const double *X, const double *y,
                        size_t n_samples, size_t n_features);

/**
 * @brief Predict the label or target for a single sample.
 *
 * @param model Fitted model. Must not be NULL.
 * @param x Feature vector (n_features). Must not be NULL.
 * @param n_features Number of features. Must match model->n_features.
 * @param out Output prediction. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n_samples * n_features + n_samples log n_samples) per query.
 */
yaj_ml_status_t knn_predict(const knn_model_t *model, const double *x,
                            size_t n_features, double *out);

/**
 * @brief Evaluate the model on a dataset.
 *
 * Classification: returns accuracy. Regression: returns R².
 *
 * @param model Fitted model.
 * @param X Feature matrix in row-major order.
 * @param y True labels or targets.
 * @param n_samples Number of samples.
 * @param n_features Number of features per sample.
 * @param score_out Output score. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 */
yaj_ml_status_t knn_score(const knn_model_t *model, const double *X,
                          const double *y, size_t n_samples,
                          size_t n_features, double *score_out);

/**
 * @brief Release all memory owned by a model.
 *
 * @param model Model to destroy.
 */
void knn_free(knn_model_t *model);

#endif /* YAJ_ML_KNN_H */
