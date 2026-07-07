#include "test_harness.h"

#include "yaj_ml/knn.h"

TEST(knn_init_and_free)
{
    knn_model_t model = {0};

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_TRUE(!model.fitted);

    knn_free(&model);
    ASSERT_TRUE(model.X == NULL);
}

TEST(knn_classification_1d)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {1.0, 2.0, 3.0, 10.0, 11.0, 12.0};
    const double y[] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    double pred;
    double accuracy;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    config.k = 3U;

    ASSERT_STATUS_OK(knn_fit(&model, &config, X, y, 6, 1));

    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){2.0}, 1, &pred));
    ASSERT_NEAR(0.0, pred, TEST_EPSILON);

    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){11.0}, 1, &pred));
    ASSERT_NEAR(1.0, pred, TEST_EPSILON);

    ASSERT_STATUS_OK(knn_score(&model, X, y, 6, 1, &accuracy));
    ASSERT_NEAR(1.0, accuracy, TEST_EPSILON);

    knn_free(&model);
}

TEST(knn_classification_2d)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        5.0, 5.0,
        5.0, 6.0,
        6.0, 5.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    double pred;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    config.k = 3U;

    ASSERT_STATUS_OK(knn_fit(&model, &config, X, y, 6, 2));

    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){0.1, 0.1}, 2, &pred));
    ASSERT_NEAR(0.0, pred, TEST_EPSILON);

    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){5.2, 5.1}, 2, &pred));
    ASSERT_NEAR(1.0, pred, TEST_EPSILON);

    knn_free(&model);
}

TEST(knn_k1_nearest)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {0.0, 1.0, 5.0};
    const double y[] = {10.0, 20.0, 30.0};
    double pred;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    config.k = 1U;

    ASSERT_STATUS_OK(knn_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){0.9}, 1, &pred));
    ASSERT_NEAR(20.0, pred, TEST_EPSILON);

    knn_free(&model);
}

TEST(knn_regression)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double pred;
    double r2;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    config.k = 1U;
    config.task = KNN_TASK_REGRESSION;

    ASSERT_STATUS_OK(knn_fit(&model, &config, X, y, 5, 1));
    ASSERT_STATUS_OK(knn_predict(&model, (const double[]){3.0}, 1, &pred));
    ASSERT_NEAR(3.0, pred, TEST_EPSILON);

    ASSERT_STATUS_OK(knn_score(&model, X, y, 5, 1, &r2));
    ASSERT_NEAR(1.0, r2, TEST_EPSILON);

    knn_free(&model);
}

TEST(knn_invalid_k)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {1.0, 2.0};
    const double y[] = {0.0, 1.0};

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    config.k = 5U;

    ASSERT_STATUS(knn_fit(&model, &config, X, y, 2, 1), YAJ_ML_ERR_INVALID_ARG);

    knn_free(&model);
}

TEST(knn_not_fitted)
{
    knn_model_t model = {0};
    const double x[] = {1.0};
    double out;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS(knn_predict(&model, x, 1, &out), YAJ_ML_ERR_NOT_FITTED);

    knn_free(&model);
}

TEST(knn_predict_dim_mismatch)
{
    knn_model_t model = {0};
    knn_config_t config;
    const double X[] = {1.0, 2.0, 3.0};
    const double y[] = {0.0, 0.0, 1.0};
    const double x[] = {1.0, 2.0};
    double out;

    ASSERT_STATUS_OK(knn_init(&model));
    ASSERT_STATUS_OK(knn_config_default(&config));
    ASSERT_STATUS_OK(knn_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS(knn_predict(&model, x, 2, &out), YAJ_ML_ERR_DIM);

    knn_free(&model);
}
