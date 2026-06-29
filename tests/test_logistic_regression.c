#include "test_harness.h"

#include "yaj_ml/logistic_regression.h"

TEST(logreg_init_and_free)
{
    logreg_model_t model = {0};

    ASSERT_STATUS_OK(logreg_init(&model, 2));
    ASSERT_EQ(2, (long long)model.n_features);
    ASSERT_TRUE(!model.fitted);
    ASSERT_NEAR(0.5, model.threshold, TEST_EPSILON);

    logreg_free(&model);
    ASSERT_TRUE(model.weights == NULL);
}

TEST(logreg_fit_and_predict_1d)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    /*
     * Labels: x < 0 -> 0, x >= 0 -> 1 (approximately)
     */
    const double X[] = {-2.0, -1.0, 1.0, 2.0, 3.0};
    const double y[] = {0.0, 0.0, 1.0, 1.0, 1.0};
    double proba;
    double cls;
    double accuracy;

    ASSERT_STATUS_OK(logreg_init(&model, 1));
    ASSERT_STATUS_OK(logreg_config_default(&config));
    config.max_iters = 5000U;
    config.learning_rate = 0.1;

    ASSERT_STATUS_OK(logreg_fit(&model, &config, X, y, 5, 1));

    ASSERT_STATUS_OK(logreg_predict_proba(&model, (const double[]){-3.0}, 1, &proba));
    ASSERT_TRUE(proba < 0.5);

    ASSERT_STATUS_OK(logreg_predict_proba(&model, (const double[]){3.0}, 1, &proba));
    ASSERT_TRUE(proba > 0.5);

    ASSERT_STATUS_OK(logreg_predict(&model, (const double[]){2.0}, 1, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(logreg_score(&model, X, y, 5, 1, &accuracy));
    ASSERT_NEAR(1.0, accuracy, 1e-9);

    logreg_free(&model);
}

TEST(logreg_fit_and_gate)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};
    double accuracy;
    double cls;

    ASSERT_STATUS_OK(logreg_init(&model, 2));
    ASSERT_STATUS_OK(logreg_config_default(&config));
    config.max_iters = 10000U;
    config.learning_rate = 0.5;

    ASSERT_STATUS_OK(logreg_fit(&model, &config, X, y, 4, 2));
    ASSERT_STATUS_OK(logreg_score(&model, X, y, 4, 2, &accuracy));
    ASSERT_NEAR(1.0, accuracy, 1e-9);

    ASSERT_STATUS_OK(logreg_predict(&model, (const double[]){1.0, 1.0}, 2, &cls));
    ASSERT_NEAR(1.0, cls, TEST_EPSILON);

    ASSERT_STATUS_OK(logreg_predict(&model, (const double[]){0.0, 1.0}, 2, &cls));
    ASSERT_NEAR(0.0, cls, TEST_EPSILON);

    logreg_free(&model);
}

TEST(logreg_invalid_labels)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    const double X[] = {1.0, 2.0};
    const double y[] = {0.5, 1.0};

    ASSERT_STATUS_OK(logreg_init(&model, 1));
    ASSERT_STATUS_OK(logreg_config_default(&config));
    ASSERT_STATUS(logreg_fit(&model, &config, X, y, 2, 1),
                  YAJ_ML_ERR_INVALID_ARG);

    logreg_free(&model);
}

TEST(logreg_not_fitted)
{
    logreg_model_t model = {0};
    const double x[] = {1.0};
    double out;

    ASSERT_STATUS_OK(logreg_init(&model, 1));
    ASSERT_STATUS(logreg_predict(&model, x, 1, &out), YAJ_ML_ERR_NOT_FITTED);

    logreg_free(&model);
}

TEST(logreg_predict_dim_mismatch)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    const double X[] = {-1.0, 1.0, 2.0};
    const double y[] = {0.0, 1.0, 1.0};
    const double x[] = {1.0, 2.0};
    double out;

    ASSERT_STATUS_OK(logreg_init(&model, 1));
    ASSERT_STATUS_OK(logreg_config_default(&config));
    ASSERT_STATUS_OK(logreg_fit(&model, &config, X, y, 3, 1));
    ASSERT_STATUS(logreg_predict(&model, x, 2, &out), YAJ_ML_ERR_DIM);

    logreg_free(&model);
}
