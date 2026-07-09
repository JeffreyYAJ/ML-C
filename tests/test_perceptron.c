#include "test_harness.h"

#include "yaj_ml/perceptron.h"

TEST(perc_init_and_free)
{
    perc_model_t model = {0};

    ASSERT_STATUS_OK(perc_init(&model, 2));
    ASSERT_EQ(2, (long long)model.n_features);
    ASSERT_TRUE(!model.fitted);

    perc_free(&model);
    ASSERT_TRUE(model.weights == NULL);
}

TEST(perc_fit_and_gate)
{
    perc_model_t model = {0};
    perc_config_t config;
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};
    double accuracy;
    double cls;

    ASSERT_STATUS_OK(perc_init(&model, 2));
    ASSERT_STATUS_OK(perc_config_default(&config));

    ASSERT_STATUS_OK(perc_fit(&model, &config, X, y, 4, 2));
    ASSERT_TRUE(model.converged);

    ASSERT_STATUS_OK(perc_score(&model, X, y, 4, 2, &accuracy));
    ASSERT_NEAR(1.0, accuracy, TEST_EPSILON);

    ASSERT_STATUS_OK(perc_predict(&model, (const double[]){1.0, 1.0}, 2, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(perc_predict(&model, (const double[]){0.0, 1.0}, 2, &cls));
    ASSERT_NEAR(0.0, cls, TEST_EPSILON);

    perc_free(&model);
}

TEST(perc_fit_1d_separable)
{
    perc_model_t model = {0};
    perc_config_t config;
    const double X[] = {-2.0, -1.0, 1.0, 2.0};
    const double y[] = {0.0, 0.0, 1.0, 1.0};
    double cls;
    double accuracy;

    ASSERT_STATUS_OK(perc_init(&model, 1));
    ASSERT_STATUS_OK(perc_config_default(&config));

    ASSERT_STATUS_OK(perc_fit(&model, &config, X, y, 4, 1));
    ASSERT_TRUE(model.converged);

    ASSERT_STATUS_OK(perc_predict(&model, (const double[]){-1.5}, 1, &cls));
    ASSERT_NEAR(0.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(perc_predict(&model, (const double[]){1.5}, 1, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(perc_score(&model, X, y, 4, 1, &accuracy));
    ASSERT_NEAR(1.0, accuracy, TEST_EPSILON);

    perc_free(&model);
}

TEST(perc_xor_not_converged)
{
    perc_model_t model = {0};
    perc_config_t config;
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 1.0, 1.0, 0.0};
    double accuracy;

    ASSERT_STATUS_OK(perc_init(&model, 2));
    ASSERT_STATUS_OK(perc_config_default(&config));
    config.max_epochs = 50U;

    ASSERT_STATUS_OK(perc_fit(&model, &config, X, y, 4, 2));
    ASSERT_TRUE(!model.converged);

    ASSERT_STATUS_OK(perc_score(&model, X, y, 4, 2, &accuracy));
    ASSERT_TRUE(accuracy < 1.0);

    perc_free(&model);
}

TEST(perc_invalid_labels)
{
    perc_model_t model = {0};
    perc_config_t config;
    const double X[] = {1.0, 2.0};
    const double y[] = {0.5, 1.0};

    ASSERT_STATUS_OK(perc_init(&model, 1));
    ASSERT_STATUS_OK(perc_config_default(&config));
    ASSERT_STATUS(perc_fit(&model, &config, X, y, 2, 1),
                  YAJ_ML_ERR_INVALID_ARG);

    perc_free(&model);
}

TEST(perc_not_fitted)
{
    perc_model_t model = {0};
    const double x[] = {1.0};
    double out;

    ASSERT_STATUS_OK(perc_init(&model, 1));
    ASSERT_STATUS(perc_predict(&model, x, 1, &out), YAJ_ML_ERR_NOT_FITTED);

    perc_free(&model);
}

TEST(perc_predict_dim_mismatch)
{
    perc_model_t model = {0};
    perc_config_t config;
    const double X[] = {-1.0, 1.0, 2.0};
    const double y[] = {0.0, 1.0, 1.0};
    const double x[] = {1.0, 2.0};
    double out;

    ASSERT_STATUS_OK(perc_init(&model, 1));
    ASSERT_STATUS_OK(perc_config_default(&config));
    ASSERT_STATUS_OK(perc_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS(perc_predict(&model, x, 2, &out), YAJ_ML_ERR_DIM);

    perc_free(&model);
}
