#ifndef YAJ_ML_MATRIX_H
#define YAJ_ML_MATRIX_H

/**
 * @file matrix.h
 * @brief Dense matrix type and basic linear algebra operations.
 */

#include "yaj_ml/error.h"
#include "yaj_ml/types.h"
#include "yaj_ml/vector.h"

/**
 * Dense matrix stored in row-major order: element (i, j) is at data[i * cols + j].
 *
 * Memory is owned by the matrix; use mat_create/mat_free to manage it.
 * Do not free @p data directly.
 */
typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} yaj_ml_mat_t;

/**
 * @brief Allocate a matrix initialized to zero.
 *
 * @param rows Number of rows. Must be greater than zero.
 * @param cols Number of columns. Must be greater than zero.
 * @param out Output matrix. Caller owns the returned object; free with mat_free.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(rows * cols)
 */
yaj_ml_status_t mat_create(size_t rows, size_t cols, yaj_ml_mat_t *out);

/**
 * @brief Release memory owned by a matrix.
 *
 * @param mat Matrix to destroy.
 *
 * @complexity O(1)
 */
void mat_free(yaj_ml_mat_t *mat);

/**
 * @brief Create a deep copy of a matrix.
 *
 * @param src Source matrix. Must not be NULL.
 * @param out Output matrix. Caller owns the returned object; free with mat_free.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(rows * cols)
 */
yaj_ml_status_t mat_copy(const yaj_ml_mat_t *src, yaj_ml_mat_t *out);

/**
 * @brief Read element at row @p row and column @p col.
 *
 * @param mat Input matrix. Must not be NULL.
 * @param row Row index (0-based).
 * @param col Column index (0-based).
 * @param out Output value. Caller owns the storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(1)
 */
yaj_ml_status_t mat_get(const yaj_ml_mat_t *mat, size_t row, size_t col,
                        double *out);

/**
 * @brief Write element at row @p row and column @p col.
 *
 * @param mat Matrix to modify. Must not be NULL.
 * @param row Row index (0-based).
 * @param col Column index (0-based).
 * @param value Value to store.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(1)
 */
yaj_ml_status_t mat_set(yaj_ml_mat_t *mat, size_t row, size_t col, double value);

/**
 * @brief Matrix multiplication: out = a * b.
 *
 * Uses a naive O(n^3) algorithm; optimized variants may be added later.
 *
 * @param a Left operand (m x k). Must not be NULL.
 * @param b Right operand (k x n). Must not be NULL.
 * @param out Output matrix (m x n). Caller must provide pre-allocated storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(m * n * k)
 */
yaj_ml_status_t mat_mul(const yaj_ml_mat_t *a, const yaj_ml_mat_t *b,
                        yaj_ml_mat_t *out);

/**
 * @brief Transpose a matrix: out = a^T.
 *
 * @param a Input matrix (m x n). Must not be NULL.
 * @param out Output matrix (n x m). Caller must provide pre-allocated storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(m * n)
 */
yaj_ml_status_t mat_transpose(const yaj_ml_mat_t *a, yaj_ml_mat_t *out);

/**
 * @brief Matrix-vector multiplication: out = mat * vec.
 *
 * @param mat Input matrix (m x n). Must not be NULL.
 * @param vec Input vector (n). Must not be NULL.
 * @param out Output vector (m). Caller must provide pre-allocated storage.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(m * n)
 */
yaj_ml_status_t mat_vec_mul(const yaj_ml_mat_t *mat, const yaj_ml_vec_t *vec,
                            yaj_ml_vec_t *out);

/**
 * @brief Augment a matrix with a column of ones (bias/intercept term).
 *
 * Given an input matrix of shape (m x n), produces output of shape (m x (n + 1))
 * where the last column is filled with 1.0. Used by regression models to absorb
 * the intercept into a single matrix multiply.
 *
 * @param src Input matrix (m x n). Must not be NULL.
 * @param out Output matrix (m x (n + 1)). Allocated by this function; free with mat_free.
 * @return YAJ_ML_OK on success, or an error code on failure.
 *
 * @complexity O(m * n)
 */
yaj_ml_status_t mat_add_row(const yaj_ml_mat_t *src, yaj_ml_mat_t *out);

#endif /* YAJ_ML_MATRIX_H */
