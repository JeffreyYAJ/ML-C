add_library(yaj_ml_logistic_regression STATIC
    logistic_regression.c
)

target_link_libraries(yaj_ml_logistic_regression PUBLIC yaj_ml)

target_include_directories(yaj_ml_logistic_regression
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

yaj_ml_set_compiler_warnings(yaj_ml_logistic_regression)
