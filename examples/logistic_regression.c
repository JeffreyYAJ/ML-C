#include "yaj_ml/logistic_regression.h"
#include <stdio.h>

int main(void)
{
    logreg_model_t model = {0};
    logreg_config_t config;
    yaj_ml_status_t status;

    /* Porte AND : sortie 1 seulement si les deux entrées valent 1 */
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};

    status = logreg_init(&model, 2);
    if (status != YAJ_ML_OK) {
        return 1;
    }

    logreg_config_default(&config);
    config.learning_rate = 0.5;
    config.max_iters = 10000U;

    status = logreg_fit(&model, &config, X, y, 4, 2);
    if (status != YAJ_ML_OK) {
        logreg_free(&model);
        return 1;
    }

    {
        const double test_cases[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        size_t i;

        printf("Porte AND — prédictions :\n");
        for (i = 0; i < 4; ++i) {
            double proba;
            double cls;

            logreg_predict_proba(&model, test_cases[i], 2, &proba);
            logreg_predict(&model, test_cases[i], 2, &cls);
            printf("  (%g, %g) -> proba=%.3f, classe=%.0f\n",
                   test_cases[i][0], test_cases[i][1], proba, cls);
        }
    }

    logreg_free(&model);
    return 0;
}
