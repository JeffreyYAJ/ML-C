add_library(yaj_ml_knn STATIC
    knn.c
)

target_link_libraries(yaj_ml_knn PUBLIC yaj_ml)

target_include_directories(yaj_ml_knn
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
)

yaj_ml_set_compiler_warnings(yaj_ml_knn)
