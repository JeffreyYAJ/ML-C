#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "yaj_ml/error.h"

#define TEST_EPSILON 1e-9

typedef void (*test_fn_t)(void);

typedef struct {
    const char *name;
    test_fn_t fn;
} test_case_t;

#define TEST(name) void test_##name(void)

static int g_test_failed = 0;

#define ASSERT_TRUE(expr)                                                      \
    do {                                                                       \
        if (!(expr)) {                                                         \
            g_test_failed = 1;                                                 \
            fprintf(stderr, "  FAIL: %s:%d: assertion failed: %s\n",           \
                    __FILE__, __LINE__, #expr);                                \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_EQ(expected, actual)                                            \
    do {                                                                       \
        const long long _exp = (long long)(expected);                          \
        const long long _act = (long long)(actual);                            \
        if (_exp != _act) {                                                    \
            g_test_failed = 1;                                                 \
            fprintf(stderr,                                                   \
                    "  FAIL: %s:%d: expected %lld, got %lld\n",                \
                    __FILE__, __LINE__, _exp, _act);                           \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_NEAR(expected, actual, epsilon)                                 \
    do {                                                                       \
        const double _exp = (double)(expected);                                  \
        const double _act = (double)(actual);                                  \
        const double _eps = (double)(epsilon);                                 \
        if (fabs(_exp - _act) > _eps) {                                        \
            g_test_failed = 1;                                                 \
            fprintf(stderr,                                                   \
                    "  FAIL: %s:%d: expected %.12g, got %.12g "               \
                    "(epsilon %.12g)\n",                                       \
                    __FILE__, __LINE__, _exp, _act, _eps);                     \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_STATUS_OK(status)                                               \
    do {                                                                       \
        const yaj_ml_status_t _status = (status);                              \
        if (_status != YAJ_ML_OK) {                                            \
            g_test_failed = 1;                                                 \
            fprintf(stderr, "  FAIL: %s:%d: expected YAJ_ML_OK, got %s\n",     \
                    __FILE__, __LINE__, yaj_ml_status_str(_status));           \
            return;                                                            \
        }                                                                      \
    } while (0)

#define ASSERT_STATUS(status, expected)                                        \
    do {                                                                       \
        const yaj_ml_status_t _status = (status);                              \
        const yaj_ml_status_t _expected = (expected);                          \
        if (_status != _expected) {                                            \
            g_test_failed = 1;                                                 \
            fprintf(stderr,                                                   \
                    "  FAIL: %s:%d: expected %s, got %s\n",                    \
                    __FILE__, __LINE__, yaj_ml_status_str(_expected),          \
                    yaj_ml_status_str(_status));                               \
            return;                                                            \
        }                                                                      \
    } while (0)

static inline int run_tests(const test_case_t *tests, size_t count)
{
    size_t i;
    size_t passed = 0;
    size_t failed = 0;

    for (i = 0; i < count; ++i) {
        printf("  %s ... ", tests[i].name);
        fflush(stdout);

        g_test_failed = 0;
        tests[i].fn();

        if (g_test_failed) {
            printf("FAIL\n");
            ++failed;
        } else {
            printf("ok\n");
            ++passed;
        }
    }

    printf("\n%zu tests passed", passed);
    if (failed > 0) {
        printf(", %zu tests failed", failed);
    }
    printf("\n");

    return failed > 0 ? 1 : 0;
}

#endif /* TEST_HARNESS_H */
