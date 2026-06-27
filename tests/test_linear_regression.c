#include "test_harness.h"

#include "yaj_ml/linear_regression.h"

TEST(lr_init_and_free)
{
    lr_model_t model = {0};

    ASSERT_STATUS_OK(lr_init(&model, 2));
    ASSERT_EQ(2, (long long)model.n_features);
    ASSERT_TRUE(!model.fitted);

    lr_free(&model);
    ASSERT_TRUE(model.weights == NULL);
}

TEST(lr_fit_normal_equation_1d)
{
    lr_model_t model = {0};
    lr_config_t config;
    /*
     * y = 2*x + 1
     * X = [1, 2, 3, 4, 5], y = [3, 5, 7, 9, 11]
     */
    const double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[] = {3.0, 5.0, 7.0, 9.0, 11.0};
    double prediction;
    double r2;

    ASSERT_STATUS_OK(lr_init(&model, 1));
    ASSERT_STATUS_OK(lr_config_default(LR_METHOD_NORMAL_EQUATION, &config));
    ASSERT_STATUS_OK(lr_fit(&model, &config, X, y, 5, 1));

    ASSERT_NEAR(2.0, model.weights[0], 1e-9);
    ASSERT_NEAR(1.0, model.weights[1], 1e-9);

    ASSERT_STATUS_OK(lr_predict(&model, &X[2], 1, &prediction));
    ASSERT_NEAR(7.0, prediction, 1e-9);

    ASSERT_STATUS_OK(lr_score(&model, X, y, 5, 1, &r2));
    ASSERT_NEAR(1.0, r2, 1e-9);

    lr_free(&model);
}

TEST(lr_fit_normal_equation_2d)
{
    lr_model_t model = {0};
    lr_config_t config;
    /*
     * y = 2*x0 + 3*x1 + 1
     */
    const double X[] = {
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        1.0, 2.0,
    };
    const double y[] = {3.0, 4.0, 6.0, 8.0, 9.0};

    ASSERT_STATUS_OK(lr_init(&model, 2));
    ASSERT_STATUS_OK(lr_config_default(LR_METHOD_NORMAL_EQUATION, &config));
    ASSERT_STATUS_OK(lr_fit(&model, &config, X, y, 5, 2));

    ASSERT_NEAR(2.0, model.weights[0], 1e-9);
    ASSERT_NEAR(3.0, model.weights[1], 1e-9);
    ASSERT_NEAR(1.0, model.weights[2], 1e-9);

    lr_free(&model);
}

TEST(lr_fit_gradient_descent)
{
    lr_model_t model = {0};
    lr_config_t config;
    const double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    const double y[] = {3.0, 5.0, 7.0, 9.0, 11.0};
    double r2;

    ASSERT_STATUS_OK(lr_init(&model, 1));
    ASSERT_STATUS_OK(lr_config_default(LR_METHOD_GRADIENT_DESCENT, &config));
    config.learning_rate = 0.05;
    config.max_iters = 5000U;
    config.tolerance = 1e-10;

    ASSERT_STATUS_OK(lr_fit(&model, &config, X, y, 5, 1));

    ASSERT_NEAR(2.0, model.weights[0], 1e-4);
    ASSERT_NEAR(1.0, model.weights[1], 1e-4);

    ASSERT_STATUS_OK(lr_score(&model, X, y, 5, 1, &r2));
    ASSERT_NEAR(1.0, r2, 1e-4);

    lr_free(&model);
}

TEST(lr_not_fitted)
{
    lr_model_t model = {0};
    const double x[] = {1.0};
    double out;

    ASSERT_STATUS_OK(lr_init(&model, 1));
    ASSERT_STATUS(lr_predict(&model, x, 1, &out), YAJ_ML_ERR_NOT_FITTED);

    lr_free(&model);
}

TEST(lr_predict_dim_mismatch)
{
    lr_model_t model = {0};
    lr_config_t config;
    const double X[] = {1.0, 2.0, 3.0};
    const double y[] = {3.0, 5.0, 7.0};
    const double x[] = {1.0, 2.0};
    double out;

    ASSERT_STATUS_OK(lr_init(&model, 1));
    ASSERT_STATUS_OK(lr_config_default(LR_METHOD_NORMAL_EQUATION, &config));
    ASSERT_STATUS_OK(lr_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS(lr_predict(&model, x, 2, &out), YAJ_ML_ERR_DIM);

    lr_free(&model);
}
