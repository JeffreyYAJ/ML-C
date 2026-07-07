#include "yaj_ml/knn.h"
#include <stdio.h>

int main(void)
{
    knn_model_t model = {0};
    knn_config_t config;
    yaj_ml_status_t status;

    /* Deux groupes de points en 2D : classe 0 près de l'origine, classe 1 près de (5,5) */
    const double X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        5.0, 5.0,
        5.0, 6.0,
        6.0, 5.0,
    };
    const double y[] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    status = knn_init(&model);
    if (status != YAJ_ML_OK) {
        return 1;
    }

    knn_config_default(&config);
    config.k = 3U;

    status = knn_fit(&model, &config, X, y, 6, 2);
    if (status != YAJ_ML_OK) {
        knn_free(&model);
        return 1;
    }

    {
        const double queries[][2] = {{0.2, 0.1}, {5.1, 5.2}, {2.5, 2.5}};
        size_t i;

        printf("KNN (k=%zu) — prédictions :\n", config.k);
        for (i = 0; i < 3; ++i) {
            double pred;
            knn_predict(&model, queries[i], 2, &pred);
            printf("  (%.1f, %.1f) -> classe %.0f\n",
                   queries[i][0], queries[i][1], pred);
        }
    }

    knn_free(&model);
    return 0;
}
