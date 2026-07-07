/**
 * @file knn.c
 * @brief k-Nearest Neighbors implementation.
 *
 * Algorithm reference: docs/fr/08_knn.md
 */

#include "yaj_ml/knn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t index;
    double distance;
} knn_neighbor_t;

static double knn_euclidean_distance(const double *a, const double *b,
                                       size_t n_features)
{
    size_t i;
    double sum_sq;
    double diff;

    /*
     * Euclidean distance: d(a,b) = sqrt(sum((a_i - b_i)^2)).
     * Same metric as vec_norm_l2 applied to (a - b).
     */
    sum_sq = 0.0;
    for (i = 0; i < n_features; ++i) {
        diff = a[i] - b[i];
        sum_sq += diff * diff;
    }

    return sqrt(sum_sq);
}

static int knn_compare_neighbors(const void *a, const void *b)
{
    const knn_neighbor_t *left = (const knn_neighbor_t *)a;
    const knn_neighbor_t *right = (const knn_neighbor_t *)b;

    if (left->distance < right->distance) {
        return -1;
    }
    if (left->distance > right->distance) {
        return 1;
    }
    return 0;
}

static yaj_ml_status_t knn_predict_classification(const knn_model_t *model,
                                                  const knn_neighbor_t *neighbors,
                                                  double *out)
{
    size_t i;
    size_t j;
    size_t best_count;
    size_t count;
    double best_label;
    double label;

    best_count = 0U;
    best_label = 0.0;

    /*
     * Majority vote among the k nearest neighbors.
     * Ties are broken by choosing the smallest label value (deterministic).
     */
    for (i = 0; i < model->k; ++i) {
        label = model->y[neighbors[i].index];
        count = 0U;

        for (j = 0; j < model->k; ++j) {
            if (model->y[neighbors[j].index] == label) {
                ++count;
            }
        }

        if (count > best_count || (count == best_count && label < best_label)) {
            best_count = count;
            best_label = label;
        }
    }

    *out = best_label;
    return YAJ_ML_OK;
}

static yaj_ml_status_t knn_predict_regression(const knn_model_t *model,
                                              const knn_neighbor_t *neighbors,
                                              double *out)
{
    size_t i;
    double sum;

    sum = 0.0;
    for (i = 0; i < model->k; ++i) {
        sum += model->y[neighbors[i].index];
    }

    *out = sum / (double)model->k;
    return YAJ_ML_OK;
}

static yaj_ml_status_t knn_find_neighbors(const knn_model_t *model,
                                        const double *x,
                                        knn_neighbor_t *neighbors)
{
    size_t i;
    const double *row;

    for (i = 0; i < model->n_samples; ++i) {
        row = model->X + i * model->n_features;
        neighbors[i].index = i;
        neighbors[i].distance = knn_euclidean_distance(x, row, model->n_features);
    }

    qsort(neighbors, model->n_samples, sizeof(knn_neighbor_t),
          knn_compare_neighbors);
    return YAJ_ML_OK;
}

yaj_ml_status_t knn_init(knn_model_t *model)
{
    if (model == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    model->X = NULL;
    model->y = NULL;
    model->n_samples = 0U;
    model->n_features = 0U;
    model->k = 0U;
    model->task = KNN_TASK_CLASSIFICATION;
    model->fitted = false;
    return YAJ_ML_OK;
}

yaj_ml_status_t knn_config_default(knn_config_t *out)
{
    if (out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    out->k = 3U;
    out->task = KNN_TASK_CLASSIFICATION;
    return YAJ_ML_OK;
}

yaj_ml_status_t knn_fit(knn_model_t *model, const knn_config_t *config,
                        const double *X, const double *y,
                        size_t n_samples, size_t n_features)
{
    knn_config_t default_config;
    size_t x_count;
    double *x_copy;
    double *y_copy;

    if (model == NULL || X == NULL || y == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (n_samples == 0U || n_features == 0U) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (config == NULL) {
        knn_config_default(&default_config);
        config = &default_config;
    }

    if (config->k == 0U || config->k > n_samples) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    knn_free(model);

    x_count = n_samples * n_features;
    x_copy = malloc(x_count * sizeof(double));
    if (x_copy == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    y_copy = malloc(n_samples * sizeof(double));
    if (y_copy == NULL) {
        free(x_copy);
        return YAJ_ML_ERR_ALLOC;
    }

    memcpy(x_copy, X, x_count * sizeof(double));
    memcpy(y_copy, y, n_samples * sizeof(double));

    model->X = x_copy;
    model->y = y_copy;
    model->n_samples = n_samples;
    model->n_features = n_features;
    model->k = config->k;
    model->task = config->task;
    model->fitted = true;
    return YAJ_ML_OK;
}

yaj_ml_status_t knn_predict(const knn_model_t *model, const double *x,
                            size_t n_features, double *out)
{
    knn_neighbor_t *neighbors;
    yaj_ml_status_t status;

    if (model == NULL || x == NULL || out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_features != model->n_features) {
        return YAJ_ML_ERR_DIM;
    }

    neighbors = malloc(model->n_samples * sizeof(knn_neighbor_t));
    if (neighbors == NULL) {
        return YAJ_ML_ERR_ALLOC;
    }

    status = knn_find_neighbors(model, x, neighbors);
    if (status != YAJ_ML_OK) {
        free(neighbors);
        return status;
    }

    if (model->task == KNN_TASK_CLASSIFICATION) {
        status = knn_predict_classification(model, neighbors, out);
    } else {
        status = knn_predict_regression(model, neighbors, out);
    }

    free(neighbors);
    return status;
}

yaj_ml_status_t knn_score(const knn_model_t *model, const double *X,
                          const double *y, size_t n_samples,
                          size_t n_features, double *score_out)
{
    yaj_ml_status_t status;
    size_t i;
    size_t correct;
    const double *row;
    double prediction;
    double y_mean;
    double ss_tot;
    double ss_res;

    if (model == NULL || X == NULL || y == NULL || score_out == NULL) {
        return YAJ_ML_ERR_NULL_PTR;
    }

    if (!model->fitted) {
        return YAJ_ML_ERR_NOT_FITTED;
    }

    if (n_samples == 0U || n_features != model->n_features) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    if (model->task == KNN_TASK_CLASSIFICATION) {
        correct = 0U;
        for (i = 0; i < n_samples; ++i) {
            row = X + i * n_features;
            status = knn_predict(model, row, n_features, &prediction);
            if (status != YAJ_ML_OK) {
                return status;
            }

            if (prediction == y[i]) {
                ++correct;
            }
        }

        *score_out = (double)correct / (double)n_samples;
        return YAJ_ML_OK;
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
        status = knn_predict(model, row, n_features, &prediction);
        if (status != YAJ_ML_OK) {
            return status;
        }

        ss_res += (y[i] - prediction) * (y[i] - prediction);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }

    if (ss_tot < 1e-15) {
        return YAJ_ML_ERR_INVALID_ARG;
    }

    *score_out = 1.0 - (ss_res / ss_tot);
    return YAJ_ML_OK;
}

void knn_free(knn_model_t *model)
{
    if (model == NULL) {
        return;
    }

    free(model->X);
    free(model->y);
    model->X = NULL;
    model->y = NULL;
    model->n_samples = 0U;
    model->n_features = 0U;
    model->k = 0U;
    model->fitted = false;
}
