#include <stdio.h>

#include "test_harness.h"

void test_status_str_ok(void);
void test_status_str_null_ptr(void);
void test_status_str_unknown(void);

void test_vec_create_and_free(void);
void test_vec_create_invalid_n(void);
void test_vec_dot_product(void);
void test_vec_dot_dim_mismatch(void);
void test_vec_add_sub(void);
void test_vec_norm_l2(void);
void test_vec_copy_and_fill(void);
void test_vec_scale(void);
void test_vec_set_zero(void);

void test_mat_create_and_free(void);
void test_mat_get_set(void);
void test_mat_mul(void);
void test_mat_transpose(void);
void test_mat_vec_mul(void);
void test_mat_add_row(void);
void test_mat_create_from_buffer(void);
void test_mat_solve_identity(void);
void test_mat_solve_2x2(void);

void test_lr_init_and_free(void);
void test_lr_fit_normal_equation_1d(void);
void test_lr_fit_normal_equation_2d(void);
void test_lr_fit_gradient_descent(void);
void test_lr_not_fitted(void);
void test_lr_predict_dim_mismatch(void);

void test_logreg_init_and_free(void);
void test_logreg_fit_and_predict_1d(void);
void test_logreg_fit_and_gate(void);
void test_logreg_invalid_labels(void);
void test_logreg_not_fitted(void);
void test_logreg_predict_dim_mismatch(void);

void test_knn_init_and_free(void);
void test_knn_classification_1d(void);
void test_knn_classification_2d(void);
void test_knn_k1_nearest(void);
void test_knn_regression(void);
void test_knn_invalid_k(void);
void test_knn_not_fitted(void);
void test_knn_predict_dim_mismatch(void);

void test_perc_init_and_free(void);
void test_perc_fit_and_gate(void);
void test_perc_fit_1d_separable(void);
void test_perc_xor_not_converged(void);
void test_perc_invalid_labels(void);
void test_perc_not_fitted(void);
void test_perc_predict_dim_mismatch(void);

void test_svm_init_and_free(void);
void test_svm_fit_1d_separable(void);
void test_svm_fit_and_gate(void);
void test_svm_invalid_labels(void);
void test_svm_not_fitted(void);
void test_svm_predict_dim_mismatch(void);

static test_case_t g_tests[] = {
    {"status_str_ok", test_status_str_ok},
    {"status_str_null_ptr", test_status_str_null_ptr},
    {"status_str_unknown", test_status_str_unknown},
    {"vec_create_and_free", test_vec_create_and_free},
    {"vec_create_invalid_n", test_vec_create_invalid_n},
    {"vec_dot_product", test_vec_dot_product},
    {"vec_dot_dim_mismatch", test_vec_dot_dim_mismatch},
    {"vec_add_sub", test_vec_add_sub},
    {"vec_norm_l2", test_vec_norm_l2},
    {"vec_copy_and_fill", test_vec_copy_and_fill},
    {"vec_scale", test_vec_scale},
    {"vec_set_zero", test_vec_set_zero},
    {"mat_create_and_free", test_mat_create_and_free},
    {"mat_get_set", test_mat_get_set},
    {"mat_mul", test_mat_mul},
    {"mat_transpose", test_mat_transpose},
    {"mat_vec_mul", test_mat_vec_mul},
    {"mat_add_row", test_mat_add_row},
    {"mat_create_from_buffer", test_mat_create_from_buffer},
    {"mat_solve_identity", test_mat_solve_identity},
    {"mat_solve_2x2", test_mat_solve_2x2},
    {"lr_init_and_free", test_lr_init_and_free},
    {"lr_fit_normal_equation_1d", test_lr_fit_normal_equation_1d},
    {"lr_fit_normal_equation_2d", test_lr_fit_normal_equation_2d},
    {"lr_fit_gradient_descent", test_lr_fit_gradient_descent},
    {"lr_not_fitted", test_lr_not_fitted},
    {"lr_predict_dim_mismatch", test_lr_predict_dim_mismatch},
    {"logreg_init_and_free", test_logreg_init_and_free},
    {"logreg_fit_and_predict_1d", test_logreg_fit_and_predict_1d},
    {"logreg_fit_and_gate", test_logreg_fit_and_gate},
    {"logreg_invalid_labels", test_logreg_invalid_labels},
    {"logreg_not_fitted", test_logreg_not_fitted},
    {"logreg_predict_dim_mismatch", test_logreg_predict_dim_mismatch},
    {"knn_init_and_free", test_knn_init_and_free},
    {"knn_classification_1d", test_knn_classification_1d},
    {"knn_classification_2d", test_knn_classification_2d},
    {"knn_k1_nearest", test_knn_k1_nearest},
    {"knn_regression", test_knn_regression},
    {"knn_invalid_k", test_knn_invalid_k},
    {"knn_not_fitted", test_knn_not_fitted},
    {"knn_predict_dim_mismatch", test_knn_predict_dim_mismatch},
    {"perc_init_and_free", test_perc_init_and_free},
    {"perc_fit_and_gate", test_perc_fit_and_gate},
    {"perc_fit_1d_separable", test_perc_fit_1d_separable},
    {"perc_xor_not_converged", test_perc_xor_not_converged},
    {"perc_invalid_labels", test_perc_invalid_labels},
    {"perc_not_fitted", test_perc_not_fitted},
    {"perc_predict_dim_mismatch", test_perc_predict_dim_mismatch},
    {"svm_init_and_free", test_svm_init_and_free},
    {"svm_fit_1d_separable", test_svm_fit_1d_separable},
    {"svm_fit_and_gate", test_svm_fit_and_gate},
    {"svm_invalid_labels", test_svm_invalid_labels},
    {"svm_not_fitted", test_svm_not_fitted},
    {"svm_predict_dim_mismatch", test_svm_predict_dim_mismatch},
};

int main(void)
{
    size_t count = sizeof(g_tests) / sizeof(g_tests[0]);
    int result;

    printf("Running %zu tests...\n\n", count);
    result = run_tests(g_tests, count);
    return result;
}
