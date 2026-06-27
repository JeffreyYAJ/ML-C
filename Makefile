# YAJ-ML — Makefile (alternative simple à CMake)
#
# Usage :
#   make          → compile en mode debug + lance les tests
#   make release  → compile optimisé
#   make test     → lance les tests
#   make clean    → supprime les fichiers compilés
#   make help     → affiche l'aide
#
# Voir docs/fr/02_makefile.md pour une explication détaillée.

CC       := gcc
AR       := ar

BUILD    := build-make
OBJ_DIR  := $(BUILD)/obj
LIB_DIR  := $(BUILD)/lib
BIN_DIR  := $(BUILD)/bin

CFLAGS   := -std=c17 -Wall -Wextra -Wpedantic -Wshadow \
            -Wconversion -Wdouble-promotion -Iinclude
LDFLAGS  := -lm

# --- sources de la bibliothèque core ---
LIB_SRCS := src/error.c src/vector.c src/matrix.c
LIB_OBJS := $(LIB_SRCS:src/%.c=$(OBJ_DIR)/%.o)
LIB      := $(LIB_DIR)/libyaj_ml.a

# --- sources des modèles ---
LR_SRCS  := models/linear_regression/linear_regression.c
LR_OBJS  := $(OBJ_DIR)/linear_regression.o
LR_LIB   := $(LIB_DIR)/libyaj_ml_linear_regression.a

# --- sources des tests ---
TEST_SRCS := tests/test_main.c tests/test_error.c \
             tests/test_vector.c tests/test_matrix.c \
             tests/test_linear_regression.c
TEST_OBJS := $(TEST_SRCS:tests/%.c=$(OBJ_DIR)/test_%.o)
TEST_BIN  := $(BIN_DIR)/test_runner

# Mode debug par défaut (-g -O0), release via `make release`
BUILD_TYPE ?= debug

ifeq ($(BUILD_TYPE),release)
    CFLAGS += -O2
else
    CFLAGS += -g -O0
endif

.PHONY: all debug release test clean help

all: debug

debug:
	$(MAKE) BUILD_TYPE=debug $(LIB) $(LR_LIB) $(TEST_BIN)
	@echo ""
	@echo "=== Build debug terminé ==="
	@echo "Bibliothèque : $(LIB)"
	@echo "Tests        : $(TEST_BIN)"

release:
	$(MAKE) BUILD_TYPE=release $(LIB) $(LR_LIB) $(TEST_BIN)
	@echo ""
	@echo "=== Build release terminé ==="

test: debug
	@echo ""
	@echo "=== Lancement des tests ==="
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD)
	@echo "Répertoire $(BUILD)/ supprimé."

help:
	@echo "Cibles disponibles :"
	@echo "  make / make debug   Compile en mode debug (-g -O0)"
	@echo "  make release        Compile en mode release (-O2)"
	@echo "  make test           Compile puis lance test_runner"
	@echo "  make clean          Supprime build-make/"
	@echo ""
	@echo "Documentation : docs/fr/README.md"

# --- règles de compilation ---

$(LIB): $(LIB_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(LR_LIB): $(LR_OBJS) | $(LIB_DIR)
	$(AR) rcs $@ $^

$(TEST_BIN): $(TEST_OBJS) $(LR_LIB) $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LR_LIB) $(LIB) $(LDFLAGS)

$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/linear_regression.o: models/linear_regression/linear_regression.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/test_%.o: tests/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -Itests -c $< -o $@

$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@
