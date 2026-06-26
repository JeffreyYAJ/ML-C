#include "yaj_ml/error.h"

const char *yaj_ml_status_str(yaj_ml_status_t status)
{
    switch (status) {
    case YAJ_ML_OK:
        return "ok";
    case YAJ_ML_ERR_NULL_PTR:
        return "null pointer";
    case YAJ_ML_ERR_ALLOC:
        return "allocation failed";
    case YAJ_ML_ERR_DIM:
        return "dimension mismatch";
    case YAJ_ML_ERR_INVALID_ARG:
        return "invalid argument";
    case YAJ_ML_ERR_NOT_FITTED:
        return "model not fitted";
    default:
        return "unknown error";
    }
}
