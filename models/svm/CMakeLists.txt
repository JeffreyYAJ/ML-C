add_library(yaj_ml_svm STATIC
    svm.c
)

target_link_libraries(yaj_ml_svm PUBLIC yaj_ml)

target_include_directories(yaj_ml_svm
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

yaj_ml_set_compiler_warnings(yaj_ml_svm)
