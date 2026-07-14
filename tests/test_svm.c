#include "test_harness.h"

#include "yaj_ml/svm.h"

TEST(svm_init_and_free)
{
    svm_model_t model = {0};

    ASSERT_STATUS_OK(svm_init(&model, 2));
    ASSERT_EQ(2, (long long)model.n_features);
    ASSERT_TRUE(!model.fitted);

    svm_free(&model);
    ASSERT_TRUE(model.weights == NULL);
}

TEST(svm_fit_1d_separable)
{
    svm_model_t model = {0};
    svm_config_t config;
    const double X[] = {-2.0, -1.0, 1.0, 2.0, 3.0};
    const double y[] = {0.0, 0.0, 1.0, 1.0, 1.0};
    double cls;
    double accuracy;

    ASSERT_STATUS_OK(svm_init(&model, 1));
    ASSERT_STATUS_OK(svm_config_default(&config));
    config.learning_rate = 0.05;
    config.max_iters = 5000U;

    ASSERT_STATUS_OK(svm_fit(&model, &config, X, y, 5, 1));

    ASSERT_STATUS_OK(svm_predict(&model, (const double[]){-1.5}, 1, &cls));
    ASSERT_NEAR(0.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(svm_predict(&model, (const double[]){2.0}, 1, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(svm_score(&model, X, y, 5, 1, &accuracy));
    ASSERT_NEAR(1.0, accuracy, 1e-6);

    svm_free(&model);
}

TEST(svm_fit_and_gate)
{
    svm_model_t model = {0};
    svm_config_t config;
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};
    double accuracy;
    double cls;

    ASSERT_STATUS_OK(svm_init(&model, 2));
    ASSERT_STATUS_OK(svm_config_default(&config));
    config.learning_rate = 0.1;
    config.lambda = 0.001;
    config.max_iters = 5000U;

    ASSERT_STATUS_OK(svm_fit(&model, &config, X, y, 4, 2));
    ASSERT_STATUS_OK(svm_score(&model, X, y, 4, 2, &accuracy));
    ASSERT_NEAR(1.0, accuracy, 1e-6);

    ASSERT_STATUS_OK(svm_predict(&model, (const double[]){1.0, 1.0}, 2, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(svm_predict(&model, (const double[]){0.0, 0.0}, 2, &cls));
    ASSERT_NEAR(0.0, cls, TEST_EPSILON);

    svm_free(&model);
}

TEST(svm_invalid_labels)
{
    svm_model_t model = {0};
    svm_config_t config;
    const double X[] = {1.0, 2.0};
    const double y[] = {0.5, 1.0};

    ASSERT_STATUS_OK(svm_init(&model, 1));
    ASSERT_STATUS_OK(svm_config_default(&config));
    ASSERT_STATUS(svm_fit(&model, &config, X, y, 2, 1),
                  YAJ_ML_ERR_INVALID_ARG);

    svm_free(&model);
}

TEST(svm_not_fitted)
{
    svm_model_t model = {0};
    const double x[] = {1.0};
    double out;

    ASSERT_STATUS_OK(svm_init(&model, 1));
    ASSERT_STATUS(svm_predict(&model, x, 1, &out), YAJ_ML_ERR_NOT_FITTED);

    svm_free(&model);
}

TEST(svm_predict_dim_mismatch)
{
    svm_model_t model = {0};
    svm_config_t config;
    const double X[] = {-1.0, 1.0, 2.0};
    const double y[] = {0.0, 1.0, 1.0};
    const double x[] = {1.0, 2.0};
    double out;

    ASSERT_STATUS_OK(svm_init(&model, 1));
    ASSERT_STATUS_OK(svm_config_default(&config));
    ASSERT_STATUS_OK(svm_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS(svm_predict(&model, x, 2, &out), YAJ_ML_ERR_DIM);

    svm_free(&model);
}
