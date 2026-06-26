#include <string.h>

#include "test_harness.h"

#include "yaj_ml/error.h"

TEST(status_str_ok)
{
    ASSERT_TRUE(strcmp(yaj_ml_status_str(YAJ_ML_OK), "ok") == 0);
}

TEST(status_str_null_ptr)
{
    ASSERT_TRUE(strcmp(yaj_ml_status_str(YAJ_ML_ERR_NULL_PTR), "null pointer") == 0);
}

TEST(status_str_unknown)
{
    ASSERT_TRUE(strcmp(yaj_ml_status_str((yaj_ml_status_t)999), "unknown error") == 0);
}
