# Le Makefile expliqué

Le **Makefile** est un fichier qui dit à l'outil `make` comment compiler ton projet. C'est l'option la **plus simple** pour YAJ-ML : pas de configuration, pas de fichiers générés compliqués.

## Prérequis

- `gcc` (ou `clang`)
- `make`
- `ar` (archiver, pour créer la bibliothèque statique)

Vérifier :

```bash
gcc --version
make --version
```

## Commandes essentielles

```bash
make          # compile en debug + lance les tests
make release  # compile optimisé (-O2)
make test     # compile puis exécute test_runner
make clean    # supprime build-make/
make help     # affiche l'aide
```

## Que se passe-t-il quand tu tapes `make` ?

```
make
  └─► make debug
        └─► compile src/*.c → build-make/obj/*.o
        └─► ar rcs → build-make/lib/libyaj_ml.a
        └─► compile tests/*.c + lie libyaj_ml.a
        └─► build-make/bin/test_runner
```

## Anatomie du Makefile

Voici le [`Makefile`](../../Makefile) annoté section par section.

### Variables de configuration

```makefile
CC       := gcc          # Compilateur C
AR       := ar           # Outil pour créer des archives .a

BUILD    := build-make   # Dossier de sortie (séparé de CMake)
OBJ_DIR  := $(BUILD)/obj
LIB_DIR  := $(BUILD)/lib
BIN_DIR  := $(BUILD)/bin

CFLAGS   := -std=c17 -Wall -Wextra -Wpedantic ...
LDFLAGS  := -lm          # Lie la bibliothèque math (sqrt, etc.)
```

| Flag | Signification |
|------|---------------|
| `-std=c17` | Utilise le standard C17 |
| `-Wall -Wextra -Wpedantic` | Active un maximum d'avertissements |
| `-Iinclude` | Cherche les headers dans `include/` |
| `-g -O0` | Mode debug : symboles de debug, pas d'optimisation |
| `-O2` | Mode release : optimisations |
| `-lm` | Lie `libm` (mathématiques) |

### Sources et objets

```makefile
LIB_SRCS := src/error.c src/vector.c src/matrix.c
LIB_OBJS := $(LIB_SRCS:src/%.c=$(OBJ_DIR)/%.o)
```

La syntaxe `$(VAR:pattern=replacement)` transforme :

```
src/error.c   →  build-make/obj/error.o
src/vector.c  →  build-make/obj/vector.o
src/matrix.c  →  build-make/obj/matrix.o
```

### Cibles principales

```makefile
all: debug          # `make` sans argument appelle debug

debug:
    $(MAKE) BUILD_TYPE=debug $(LIB) $(TEST_BIN)

release:
    $(MAKE) BUILD_TYPE=release $(LIB) $(TEST_BIN)
```

Une **cible** (target) est un « objectif » à construire. `make debug` construit la bibliothèque et le binaire de test.

### Créer la bibliothèque statique

```makefile
$(LIB): $(LIB_OBJS) | $(LIB_DIR)
    $(AR) rcs $@ $^
```

| Symbole | Signification |
|---------|---------------|
| `$@` | La cible (`libyaj_ml.a`) |
| `$^` | Toutes les dépendances (les `.o`) |
| `\|` | Ordre seulement (crée le dossier avant) |
| `ar rcs` | **r**emplace, crée l'**i**ndex, ajoute les fichiers |

Une **bibliothèque statique** (`.a`) est une archive de fichiers objet. À la liaison, le linker copie le code nécessaire dans ton exécutable final.

### Lier le test runner

```makefile
$(TEST_BIN): $(TEST_OBJS) $(LIB) | $(BIN_DIR)
    $(CC) $(CFLAGS) -o $@ $(TEST_OBJS) $(LIB) $(LDFLAGS)
```

Étapes :
1. Compile chaque `tests/*.c` en `.o`
2. Lie tous les `.o` + `libyaj_ml.a` + `libm`
3. Produit `build-make/bin/test_runner`

### Règles de compilation

```makefile
$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
    $(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/test_%.o: tests/%.c | $(OBJ_DIR)
    $(CC) $(CFLAGS) -Itests -c $< -o $@
```

| Symbole | Signification |
|---------|---------------|
| `%` | Joker (n'importe quel nom) |
| `$<` | La première dépendance (le fichier `.c`) |
| `-c` | Compile seulement, ne lie pas |

**Pattern rule** : `$(OBJ_DIR)/%.o: src/%.c` signifie « pour transformer `src/foo.c` en `build-make/obj/foo.o`, exécute gcc -c ».

## Makefile vs CMake

| Aspect | Makefile | CMake |
|--------|----------|-------|
| Courbe d'apprentissage | Faible | Plus élevée |
| Portabilité multi-plateforme | Manuelle | Automatique |
| Intégration IDE | Limitée | Excellente |
| Adapté à YAJ-ML maintenant | Oui | Oui (pour CI/futur) |

**Conseil** : utilise `make` pour apprendre et itérer rapidement. Passe à CMake quand le projet grandira (plusieurs modèles, CI, etc.).

## Ajouter un nouveau fichier source

Quand tu ajoutes par exemple `src/dataset.c` :

1. Ajoute-le à `LIB_SRCS` :
   ```makefile
   LIB_SRCS := src/error.c src/vector.c src/matrix.c src/dataset.c
   ```
2. Crée `include/yaj_ml/dataset.h`
3. `make clean && make test`

C'est tout — pas de fichier de config supplémentaire.

## Dépannage

| Problème | Solution |
|----------|----------|
| `make: command not found` | Installe `build-essential` (Debian/Ubuntu) |
| Erreur de compilation | Lis le message ; souvent un header manquant ou un typo |
| Tests échouent | `./build-make/bin/test_runner` pour voir quel test |
| Veux repartir de zéro | `make clean` |
