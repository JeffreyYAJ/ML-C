#include "yaj_ml/perceptron.h"
#include <stdio.h>

int main(void)
{
    perc_model_t model = {0};
    perc_config_t config;
    yaj_ml_status_t status;

    /* Porte AND — linéairement séparable */
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0};

    status = perc_init(&model, 2);
    if (status != YAJ_ML_OK) {
        return 1;
    }

    perc_config_default(&config);
    status = perc_fit(&model, &config, X, y, 4, 2);
    if (status != YAJ_ML_OK) {
        perc_free(&model);
        return 1;
    }

    printf("Porte AND — perceptron %s\n",
           model.converged ? "convergé" : "non convergé");

    {
        const double queries[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        size_t i;

        for (i = 0; i < 4; ++i) {
            double cls;
            perc_predict(&model, queries[i], 2, &cls);
            printf("  (%g, %g) -> %.0f\n", queries[i][0], queries[i][1], cls);
        }
    }

    perc_free(&model);
    return 0;
}
