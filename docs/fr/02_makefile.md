# 02 — Comprendre le Makefile

Si tu n'as jamais utilisé CMake, **commence par le Makefile**. Il fait la même chose, mais de façon explicite et lisible.

## Qu'est-ce qu'un Makefile ?

Un Makefile est un fichier de **recettes** pour le programme `make`. Chaque recette dit :

> « Pour produire X, j'ai besoin de Y, et voici la commande à exécuter. »

`make` vérifie quels fichiers ont changé et ne recompile **que le nécessaire** (compilation incrémentale).

## Commandes de base

```bash
make          # compile en mode debug
make test     # compile + lance les tests
make release  # compile optimisé (-O2)
make clean    # supprime build-make/
make help     # affiche l'aide
```

## Anatomie du Makefile YAJ-ML

### Variables (configuration)

```makefile
CC       := gcc                          # compilateur
CFLAGS   := -std=c17 -Wall -Wextra ... # flags de compilation
BUILD    := build-make                   # dossier de sortie
LIB      := $(BUILD)/lib/libyaj_ml.a     # bibliothèque finale
TEST_BIN := $(BUILD)/bin/test_runner     # exécutable de tests
```

Les variables évitent de répéter les mêmes chemins partout.

### Liste des sources et objets

```makefile
LIB_SRCS := src/error.c src/vector.c src/matrix.c
LIB_OBJS := $(LIB_SRCS:src/%.c=$(OBJ_DIR)/%.o)
```

La syntaxe `$(VAR:src/%.c=$(OBJ_DIR)/%.o)` transforme :
- `src/error.c` → `build-make/obj/error.o`
- `src/vector.c` → `build-make/obj/vector.o`
- etc.

C'est une **substitution de pattern** Makefile.

### Cibles (targets)

Une cible = un fichier à produire + ses dépendances + la commande.

```makefile
$(LIB): $(LIB_OBJS) | $(LIB_DIR)
    $(AR) rcs $@ $^
```

Lecture :
- **Cible** : `$(LIB)` = `libyaj_ml.a`
- **Dépendances** : tous les `.o` + le dossier `lib/` doit exister
- **Commande** : `ar rcs` crée une bibliothèque statique à partir des objets

Symboles spéciaux :
| Symbole | Signification |
|---------|---------------|
| `$@` | la cible (libyaj_ml.a) |
| `$^` | toutes les dépendances (tous les .o) |
| `$<` | la première dépendance |

### Règle de compilation d'un .c → .o

```makefile
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
    $(CC) $(CFLAGS) -c $< -o $@
```

- `%` = joker (error, vector, matrix…)
- `-c` = compile seulement, ne lie pas
- `$<` = le fichier .c source
- `$@` = le fichier .o de sortie

### Règle de liaison du test runner

```makefile
$(TEST_BIN): $(TEST_OBJS) $(LIB) | $(BIN_DIR)
    $(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LIB) $(LDFLAGS)
```

Ici on **lie** les objets de test + la bibliothèque + `libm` (`-lm`) pour produire l'exécutable.

### Cibles `.PHONY`

```makefile
.PHONY: all debug release test clean help
```

Ces cibles ne produisent pas un fichier du même nom. Ce sont des **commandes** que `make` exécute toujours quand on les demande.

## Mode debug vs release

```makefile
ifeq ($(BUILD_TYPE),release)
    CFLAGS += -O2        # optimisations
else
    CFLAGS += -g -O0     # symboles debug, pas d'optimisation
endif
```

- **Debug** (`-g -O0`) : facile à déboguer avec `gdb`, compilation rapide.
- **Release** (`-O2`) : code plus rapide, plus dur à déboguer.

## Où vont les fichiers compilés ?

```
build-make/
├── obj/
│   ├── error.o
│   ├── vector.o
│   ├── matrix.o
│   ├── test_main.o
│   ├── test_error.o
│   ├── test_vector.o
│   └── test_matrix.o
├── lib/
│   └── libyaj_ml.a
└── bin/
    └── test_runner
```

Séparé du dossier `build/` de CMake pour éviter les conflits.

## Équivalence Makefile ↔ commandes manuelles

Si tu veux comprendre ce que `make` fait, voici l'équivalent à la main :

```bash
mkdir -p build-make/obj build-make/lib build-make/bin

# Compiler chaque source
gcc -std=c17 -Wall -Wextra -Wpedantic -Iinclude -g -O0 -c src/error.c   -o build-make/obj/error.o
gcc -std=c17 -Wall -Wextra -Wpedantic -Iinclude -g -O0 -c src/vector.c -o build-make/obj/vector.o
gcc -std=c17 -Wall -Wextra -Wpedantic -Iinclude -g -O0 -c src/matrix.c -o build-make/obj/matrix.o

# Créer la bibliothèque statique
ar rcs build-make/lib/libyaj_ml.a build-make/obj/error.o build-make/obj/vector.o build-make/obj/matrix.o

# Compiler les tests
gcc -std=c17 -Wall -Wextra -Wpedantic -Iinclude -Itests -g -O0 -c tests/test_main.c   -o build-make/obj/test_main.o
# ... (autres tests)

# Lier l'exécutable
gcc -g -O0 -o build-make/bin/test_runner build-make/obj/test_*.o build-make/lib/libyaj_ml.a -lm

# Lancer
./build-make/bin/test_runner
```

Le Makefile automatise exactement ça, et ne recompile que ce qui a changé.

## Makefile vs CMake : lequel utiliser ?

| | Makefile | CMake |
|---|----------|-------|
| Lisibilité | Très explicite | Indirection (génère des fichiers) |
| Portabilité | Linux/macOS facile | Windows + Linux + macOS |
| Scalabilité | Devient verbeux sur gros projets | Gère bien les gros projets |
| Pour apprendre | **Oui, commence ici** | Utile ensuite en entreprise |

Les deux coexistent dans ce projet. Tu peux n'utiliser que le Makefile si tu préfères.
