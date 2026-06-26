#include "test_harness.h"

#include "yaj_ml/matrix.h"

TEST(mat_create_and_free)
{
    yaj_ml_mat_t m;
    yaj_ml_status_t status;

    status = mat_create(2, 3, &m);
    ASSERT_STATUS_OK(status);
    ASSERT_EQ(2, m.rows);
    ASSERT_EQ(3, m.cols);
    ASSERT_NEAR(0.0, m.data[0], TEST_EPSILON);
    mat_free(&m);
    ASSERT_EQ(0, m.rows);
    ASSERT_TRUE(m.data == NULL);
}

TEST(mat_get_set)
{
    yaj_ml_mat_t m;
    double value;

    ASSERT_STATUS_OK(mat_create(2, 2, &m));
    ASSERT_STATUS_OK(mat_set(&m, 0, 1, 3.5));
    ASSERT_STATUS_OK(mat_get(&m, 0, 1, &value));
    ASSERT_NEAR(3.5, value, TEST_EPSILON);

    mat_free(&m);
}

TEST(mat_mul)
{
    yaj_ml_mat_t a;
    yaj_ml_mat_t b;
    yaj_ml_mat_t c;

    ASSERT_STATUS_OK(mat_create(2, 2, &a));
    ASSERT_STATUS_OK(mat_create(2, 2, &b));
    ASSERT_STATUS_OK(mat_create(2, 2, &c));

    ASSERT_STATUS_OK(mat_set(&a, 0, 0, 1.0));
    ASSERT_STATUS_OK(mat_set(&a, 0, 1, 2.0));
    ASSERT_STATUS_OK(mat_set(&a, 1, 0, 3.0));
    ASSERT_STATUS_OK(mat_set(&a, 1, 1, 4.0));

    ASSERT_STATUS_OK(mat_set(&b, 0, 0, 5.0));
    ASSERT_STATUS_OK(mat_set(&b, 0, 1, 6.0));
    ASSERT_STATUS_OK(mat_set(&b, 1, 0, 7.0));
    ASSERT_STATUS_OK(mat_set(&b, 1, 1, 8.0));

    ASSERT_STATUS_OK(mat_mul(&a, &b, &c));

    ASSERT_NEAR(19.0, c.data[0], TEST_EPSILON);
    ASSERT_NEAR(22.0, c.data[1], TEST_EPSILON);
    ASSERT_NEAR(43.0, c.data[2], TEST_EPSILON);
    ASSERT_NEAR(50.0, c.data[3], TEST_EPSILON);

    mat_free(&a);
    mat_free(&b);
    mat_free(&c);
}

TEST(mat_transpose)
{
    yaj_ml_mat_t a;
    yaj_ml_mat_t t;
    double value;

    ASSERT_STATUS_OK(mat_create(2, 3, &a));
    ASSERT_STATUS_OK(mat_create(3, 2, &t));

    ASSERT_STATUS_OK(mat_set(&a, 0, 0, 1.0));
    ASSERT_STATUS_OK(mat_set(&a, 0, 1, 2.0));
    ASSERT_STATUS_OK(mat_set(&a, 0, 2, 3.0));
    ASSERT_STATUS_OK(mat_set(&a, 1, 0, 4.0));
    ASSERT_STATUS_OK(mat_set(&a, 1, 1, 5.0));
    ASSERT_STATUS_OK(mat_set(&a, 1, 2, 6.0));

    ASSERT_STATUS_OK(mat_transpose(&a, &t));
    ASSERT_STATUS_OK(mat_get(&t, 0, 1, &value));
    ASSERT_NEAR(4.0, value, TEST_EPSILON);
    ASSERT_STATUS_OK(mat_get(&t, 2, 0, &value));
    ASSERT_NEAR(3.0, value, TEST_EPSILON);

    mat_free(&a);
    mat_free(&t);
}

TEST(mat_vec_mul)
{
    yaj_ml_mat_t m;
    yaj_ml_vec_t v;
    yaj_ml_vec_t out;

    ASSERT_STATUS_OK(mat_create(2, 2, &m));
    ASSERT_STATUS_OK(vec_create(2, &v));
    ASSERT_STATUS_OK(vec_create(2, &out));

    ASSERT_STATUS_OK(mat_set(&m, 0, 0, 1.0));
    ASSERT_STATUS_OK(mat_set(&m, 0, 1, 2.0));
    ASSERT_STATUS_OK(mat_set(&m, 1, 0, 3.0));
    ASSERT_STATUS_OK(mat_set(&m, 1, 1, 4.0));

    v.data[0] = 2.0;
    v.data[1] = 1.0;

    ASSERT_STATUS_OK(mat_vec_mul(&m, &v, &out));
    ASSERT_NEAR(4.0, out.data[0], TEST_EPSILON);
    ASSERT_NEAR(10.0, out.data[1], TEST_EPSILON);

    mat_free(&m);
    vec_free(&v);
    vec_free(&out);
}

TEST(mat_add_row)
{
    yaj_ml_mat_t src;
    yaj_ml_mat_t out = {0};
    double value;

    ASSERT_STATUS_OK(mat_create(2, 2, &src));
    ASSERT_STATUS_OK(mat_set(&src, 0, 0, 1.0));
    ASSERT_STATUS_OK(mat_set(&src, 0, 1, 2.0));
    ASSERT_STATUS_OK(mat_set(&src, 1, 0, 3.0));
    ASSERT_STATUS_OK(mat_set(&src, 1, 1, 4.0));

    ASSERT_STATUS_OK(mat_add_row(&src, &out));
    ASSERT_EQ(2, out.rows);
    ASSERT_EQ(3, out.cols);

    ASSERT_STATUS_OK(mat_get(&out, 0, 2, &value));
    ASSERT_NEAR(1.0, value, TEST_EPSILON);
    ASSERT_STATUS_OK(mat_get(&out, 1, 0, &value));
    ASSERT_NEAR(3.0, value, TEST_EPSILON);

    mat_free(&src);
    mat_free(&out);
}
