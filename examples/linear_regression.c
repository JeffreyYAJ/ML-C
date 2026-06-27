#include "yaj_ml/linear_regression.h"
#include <stdio.h>

int main(void)
{
    lr_model_t model = {0};
    lr_config_t config;
    yaj_ml_status_t status;

    /* 1. Initialiser — dire combien de features tu as */
    status = lr_init(&model, 1);  /* 1 feature : la surface */
    if (status != YAJ_ML_OK) return 1;

    /* 2. Préparer les données d'entraînement */
    const double X[] = {50.0, 80.0, 100.0, 120.0, 150.0};  /* surfaces m² */
    const double y[] = {150.0, 210.0, 250.0, 290.0, 350.0}; /* prix en k€ */
    size_t n_samples = 5;
    size_t n_features = 1;

    /* 3. Entraîner */
    lr_config_default(LR_METHOD_NORMAL_EQUATION, &config);
    status = lr_fit(&model, &config, X, y, n_samples, n_features);
    if (status != YAJ_ML_OK) return 1;

    /* 4. Prédire sur un nouveau cas */
    const double new_house[] = {130.0};  /* 130 m² */
    double predicted_price;

    lr_predict(&model, new_house, 1, &predicted_price);
    printf("Prix prédit pour 130 m² : %.1f k€\n", predicted_price);

    lr_free(&model);
    return 0;
}