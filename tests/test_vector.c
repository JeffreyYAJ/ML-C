#include "test_harness.h"

#include "yaj_ml/vector.h"

TEST(vec_create_and_free)
{
    yaj_ml_vec_t v;
    yaj_ml_status_t status;

    status = vec_create(3, &v);
    ASSERT_STATUS_OK(status);
    ASSERT_EQ(3, v.n);
    ASSERT_NEAR(0.0, v.data[0], TEST_EPSILON);
    vec_free(&v);
    ASSERT_EQ(0, v.n);
    ASSERT_TRUE(v.data == NULL);
}

TEST(vec_create_invalid_n)
{
    yaj_ml_vec_t v;

    ASSERT_STATUS(vec_create(0, &v), YAJ_ML_ERR_INVALID_ARG);
}

TEST(vec_dot_product)
{
    yaj_ml_vec_t a;
    yaj_ml_vec_t b;
    double result;

    ASSERT_STATUS_OK(vec_create(3, &a));
    ASSERT_STATUS_OK(vec_create(3, &b));

    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    b.data[0] = 4.0;
    b.data[1] = 5.0;
    b.data[2] = 6.0;

    ASSERT_STATUS_OK(vec_dot(&a, &b, &result));
    ASSERT_NEAR(32.0, result, TEST_EPSILON);

    vec_free(&a);
    vec_free(&b);
}

TEST(vec_dot_dim_mismatch)
{
    yaj_ml_vec_t a;
    yaj_ml_vec_t b;
    double result;

    ASSERT_STATUS_OK(vec_create(2, &a));
    ASSERT_STATUS_OK(vec_create(3, &b));
    ASSERT_STATUS(vec_dot(&a, &b, &result), YAJ_ML_ERR_DIM);

    vec_free(&a);
    vec_free(&b);
}

TEST(vec_add_sub)
{
    yaj_ml_vec_t a;
    yaj_ml_vec_t b;
    yaj_ml_vec_t sum;
    yaj_ml_vec_t diff;

    ASSERT_STATUS_OK(vec_create(2, &a));
    ASSERT_STATUS_OK(vec_create(2, &b));
    ASSERT_STATUS_OK(vec_create(2, &sum));
    ASSERT_STATUS_OK(vec_create(2, &diff));

    a.data[0] = 5.0;
    a.data[1] = 3.0;
    b.data[0] = 1.0;
    b.data[1] = 2.0;

    ASSERT_STATUS_OK(vec_add(&a, &b, &sum));
    ASSERT_NEAR(6.0, sum.data[0], TEST_EPSILON);
    ASSERT_NEAR(5.0, sum.data[1], TEST_EPSILON);

    ASSERT_STATUS_OK(vec_sub(&a, &b, &diff));
    ASSERT_NEAR(4.0, diff.data[0], TEST_EPSILON);
    ASSERT_NEAR(1.0, diff.data[1], TEST_EPSILON);

    vec_free(&a);
    vec_free(&b);
    vec_free(&sum);
    vec_free(&diff);
}

TEST(vec_norm_l2)
{
    yaj_ml_vec_t v;
    double norm;

    ASSERT_STATUS_OK(vec_create(2, &v));
    v.data[0] = 3.0;
    v.data[1] = 4.0;

    ASSERT_STATUS_OK(vec_norm_l2(&v, &norm));
    ASSERT_NEAR(5.0, norm, TEST_EPSILON);

    vec_free(&v);
}

TEST(vec_copy_and_fill)
{
    yaj_ml_vec_t src;
    yaj_ml_vec_t dst;

    ASSERT_STATUS_OK(vec_create(3, &src));
    ASSERT_STATUS_OK(vec_fill(&src, 2.5));
    ASSERT_STATUS_OK(vec_copy(&src, &dst));

    ASSERT_EQ(3, dst.n);
    ASSERT_NEAR(2.5, dst.data[0], TEST_EPSILON);
    ASSERT_NEAR(2.5, dst.data[2], TEST_EPSILON);

    vec_free(&src);
    vec_free(&dst);
}

TEST(vec_scale)
{
    yaj_ml_vec_t v;

    ASSERT_STATUS_OK(vec_create(2, &v));
    v.data[0] = 2.0;
    v.data[1] = -1.0;

    ASSERT_STATUS_OK(vec_scale(3.0, &v));
    ASSERT_NEAR(6.0, v.data[0], TEST_EPSILON);
    ASSERT_NEAR(-3.0, v.data[1], TEST_EPSILON);

    vec_free(&v);
}

TEST(vec_set_zero)
{
    yaj_ml_vec_t v;

    ASSERT_STATUS_OK(vec_create(2, &v));
    v.data[0] = 7.0;
    v.data[1] = -2.0;

    ASSERT_STATUS_OK(vec_set_zero(&v));
    ASSERT_NEAR(0.0, v.data[0], TEST_EPSILON);
    ASSERT_NEAR(0.0, v.data[1], TEST_EPSILON);

    vec_free(&v);
}
