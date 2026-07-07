add_executable(test_runner
    test_main.c
    test_error.c
    test_vector.c
    test_matrix.c
    test_linear_regression.c
    test_logistic_regression.c
    test_knn.c
)

target_link_libraries(test_runner PRIVATE
    yaj_ml
    yaj_ml_linear_regression
    yaj_ml_logistic_regression
    yaj_ml_knn
)
target_include_directories(test_runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

yaj_ml_set_compiler_warnings(test_runner)

add_test(NAME unit_tests COMMAND test_runner)
