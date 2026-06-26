#ifndef YAJ_ML_MODEL_API_H
#define YAJ_ML_MODEL_API_H

/**
 * @file model_api.h
 * @brief Naming conventions for YAJ-ML model APIs.
 *
 * Every model in YAJ-ML MUST expose a consistent set of lifecycle functions.
 * Each model defines its own struct and configuration type; there is no generic
 * vtable — this keeps the library simple and educational.
 *
 * Required functions (replace `<model>` with the model prefix, e.g. `lr` for
 * linear regression):
 *
 * @code
 * yaj_ml_status_t <model>_init(<Model> *model, ...);
 * yaj_ml_status_t <model>_fit(<Model> *model, ...);
 * yaj_ml_status_t <model>_predict(const <Model> *model, ..., double *out);
 * yaj_ml_status_t <model>_score(const <Model> *model, ..., double *out);
 * void            <model>_free(<Model> *model);
 * @endcode
 *
 * Ownership rules:
 * - `<model>_init` allocates all internal state owned by the struct.
 * - `<model>_free` releases every resource allocated by init and fit.
 * - Output buffers passed to predict/score are owned by the caller unless
 *   documented otherwise.
 * - fit must be called before predict or score; otherwise return
 *   YAJ_ML_ERR_NOT_FITTED.
 *
 * Example usage pattern:
 *
 * @code
 * lr_model_t model;
 * lr_init(&model, ...);
 * lr_fit(&model, X, y, n_samples, n_features);
 * lr_predict(&model, x, n_features, &prediction);
 * lr_score(&model, X_test, y_test, n_test, &score);
 * lr_free(&model);
 * @endcode
 */

#endif /* YAJ_ML_MODEL_API_H */
