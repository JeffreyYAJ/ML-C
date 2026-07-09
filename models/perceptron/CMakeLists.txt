add_library(yaj_ml_perceptron STATIC
    perceptron.c
)

target_link_libraries(yaj_ml_perceptron PUBLIC yaj_ml)

target_include_directories(yaj_ml_perceptron
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

yaj_ml_set_compiler_warnings(yaj_ml_perceptron)
