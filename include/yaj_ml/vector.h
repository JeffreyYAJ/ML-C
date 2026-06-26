#ifndef YAJ_ML_VECTOR_H
#define YAJ_ML_VECTOR_H

/**
 * @file vector.h
 * @brief Dense vector type and basic linear algebra operations.
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"

/**
 * Dense vector stored as a contiguous array of doubles.
 *
 * Memory is owned by the vector; use vec_create/vec_free to manage it.
 * Do not free @p data directly.
 */
typedef struct {
    size_t n;
    double *data;
} yaj_ml_vec_t;

/**
 * @brief Allocate a vector of length @p n initialized to zero.
 *
 * @param n Number of elements. Must be greater than zero.
 * @param out Output vector. Caller owns the returned object; free with vec_free.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_create(size_t n, yaj_ml_vec_t *out);

/**
 * @brief Release memory owned by a vector.
 *
 * @param vec Vector to destroy. Safe to call with NULL @p data after init to zero.
 *
 * @complexity O(1)
 */
void vec_free(yaj_ml_vec_t *vec);

/**
 * @brief Create a deep copy of a vector.
 *
 * @param src Source vector. Must not be NULL.
 * @param out Output vector. Caller owns the returned object; free with vec_free.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_copy(const yaj_ml_vec_t *src, yaj_ml_vec_t *out);

/**
 * @brief Compute the dot product of two vectors.
 *
 * @param a First vector. Must not be NULL.
 * @param b Second vector. Must not be NULL.
 * @param out Output scalar. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_dot(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b, double *out);

/**
 * @brief Element-wise addition: out = a + b.
 *
 * @param a First vector. Must not be NULL.
 * @param b Second vector. Must not be NULL.
 * @param out Output vector with same length as inputs. Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_add(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b,
                        yaj_ml_vec_t *out);

/**
 * @brief Element-wise subtraction: out = a - b.
 *
 * @param a First vector. Must not be NULL.
 * @param b Second vector. Must not be NULL.
 * @param out Output vector with same length as inputs. Caller owns storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_sub(const yaj_ml_vec_t *a, const yaj_ml_vec_t *b,
                        yaj_ml_vec_t *out);

/**
 * @brief Scale a vector in place: v = alpha * v.
 *
 * @param alpha Scalar multiplier.
 * @param v Vector to scale. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_scale(double alpha, yaj_ml_vec_t *v);

/**
 * @brief Compute the Euclidean (L2) norm of a vector.
 *
 * Uses a stable accumulation to reduce precision loss on large vectors.
 *
 * @param v Input vector. Must not be NULL.
 * @param out Output norm. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_norm_l2(const yaj_ml_vec_t *v, double *out);

/**
 * @brief Fill every element of a vector with a constant value.
 *
 * @param v Vector to fill. Must not be NULL.
 * @param value Value to assign to each element.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_fill(yaj_ml_vec_t *v, double value);

/**
 * @brief Set every element of a vector to zero.
 *
 * @param v Vector to zero. Must not be NULL.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(n)
 */
yaj_ml_status_t vec_set_zero(yaj_ml_vec_t *v);

#endif /* YAJ_ML_VECTOR_H */
