#ifndef YAJ_ML_ERROR_H
#define YAJ_ML_ERROR_H

/**
 * @file error.h
 * @brief Error codes and status reporting for YAJ-ML.
 */

/**
 * Status codes returned by all public YAJ-ML functions.
 */
typedef enum {
    YAJ_ML_OK = 0,
    YAJ_ML_ERR_NULL_PTR,
    YAJ_ML_ERR_ALLOC,
    YAJ_ML_ERR_DIM,
    YAJ_ML_ERR_INVALID_ARG,
    YAJ_ML_ERR_NOT_FITTED,
    YAJ_ML_ERR_SINGULAR,
} yaj_ml_status_t;

/**
 * @brief Return a human-readable description of a status code.
 *
 * @param status Status code to describe.
 * @return Static string describing @p status. Never NULL.
 *
 * @complexity O(1)
 */
const char *yaj_ml_status_str(yaj_ml_status_t status);

#endif /* YAJ_ML_ERROR_H */
