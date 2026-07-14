#include "yaj_ml/svm.h"
#include <stdio.h>

int main(void)
{
    svm_model_t model = {0};
    svm_config_t config;
    yaj_ml_status_t status;

    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};

    status = svm_init(&model, 2);
    if (status != YAJ_ML_OK) {
        return 1;
    }

    svm_config_default(&config);
    config.learning_rate = 0.1;
    config.lambda = 0.001;
    config.max_iters = 5000U;

    status = svm_fit(&model, &config, X, y, 4, 2);
    if (status != YAJ_ML_OK) {
        svm_free(&model);
        return 1;
    }

    {
        const double queries[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        size_t i;

        printf("SVM linéaire — prédictions (porte AND) :\n");
        for (i = 0; i < 4; ++i) {
            double cls;
            svm_predict(&model, queries[i], 2, &cls);
            printf("  (%g, %g) -> %.0f\n", queries[i][0], queries[i][1], cls);
        }
    }

    svm_free(&model);
    return 0;
}
