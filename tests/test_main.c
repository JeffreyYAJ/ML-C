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
};

int main(void)
{
    size_t count = sizeof(g_tests) / sizeof(g_tests[0]);
    int result;

    printf("Running %zu tests...\n\n", count);
    result = run_tests(g_tests, count);
    return result;
}
