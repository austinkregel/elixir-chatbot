# ==============================================================================
# Docker Image Build & Deploy
# ==============================================================================

STACK_NAME     := chatbot
REGISTRY       ?= ghcr.io/austinkregel

.PHONY: train-image-rocm train-image-cuda app-image push-images \
	login train-rocm train-cuda deploy stack-down clean-volumes

login:
	@echo $(CR_PAT) | docker login ghcr.io -u austinkregel --password-stdin

train-image-rocm:
	docker build -f Dockerfile.train.rocm -t $(REGISTRY)/chatbot-train:rocm .

train-image-cuda:
	docker build -f Dockerfile.train.cuda -t $(REGISTRY)/chatbot-train:cuda .

app-image:
	docker build -f Dockerfile -t $(REGISTRY)/chatbot:latest .

push-images:
	docker push $(REGISTRY)/chatbot-train:rocm
	docker push $(REGISTRY)/chatbot-train:cuda
	docker push $(REGISTRY)/chatbot:latest

train-rocm:
	docker service scale $(STACK_NAME)_train-rocm=1

train-cuda:
	docker service scale $(STACK_NAME)_train-cuda=1

deploy:
	docker stack deploy -c docker-stack.yml --with-registry-auth --resolve-image never $(STACK_NAME)

stack-down:
	docker stack rm $(STACK_NAME)

clean-volumes:
	@echo "Removing build cache volumes (stack must be down first)..."
	-docker volume rm $(STACK_NAME)_app_deps $(STACK_NAME)_app_build
	-docker volume rm $(STACK_NAME)_rocm_deps $(STACK_NAME)_rocm_build $(STACK_NAME)_rocm_xla_cache
	-docker volume rm $(STACK_NAME)_cuda_deps $(STACK_NAME)_cuda_build
	@echo "Build cache volumes removed. Next container start will recompile from scratch."

# ==============================================================================
# Tree-sitter Grammar Compilation
#
# Clones tree-sitter grammar repos, compiles them into shared libraries (.so),
# and installs them to the location Brain.Code.LanguageGrammar expects.
#
# Usage:
#   make grammars        # Build all grammars
#   make grammar-elixir  # Build a single grammar
#   make clean-grammars  # Remove cloned repos and compiled libs
#   make list-grammars   # Show status of all grammars
# ==============================================================================

GRAMMARS_DIR   := apps/brain/priv/code/grammars
VENDOR_DIR     := vendor/tree-sitter-grammars

CC             := gcc
CXX            := g++
CFLAGS         := -shared -fPIC -O2
CXXFLAGS       := -shared -fPIC -O2

# Grammar repos: name -> GitHub URL
# Most live under github.com/tree-sitter, elixir is under elixir-lang
GRAMMAR_REPOS := \
	tree-sitter-c=https://github.com/tree-sitter/tree-sitter-c \
	tree-sitter-cpp=https://github.com/tree-sitter/tree-sitter-cpp \
	tree-sitter-java=https://github.com/tree-sitter/tree-sitter-java \
	tree-sitter-c-sharp=https://github.com/tree-sitter/tree-sitter-c-sharp \
	tree-sitter-php=https://github.com/tree-sitter/tree-sitter-php \
	tree-sitter-python=https://github.com/tree-sitter/tree-sitter-python \
	tree-sitter-ruby=https://github.com/tree-sitter/tree-sitter-ruby \
	tree-sitter-elixir=https://github.com/elixir-lang/tree-sitter-elixir \
	tree-sitter-go=https://github.com/tree-sitter/tree-sitter-go

GRAMMAR_NAMES := tree-sitter-c tree-sitter-cpp tree-sitter-java \
	tree-sitter-c-sharp tree-sitter-php tree-sitter-python \
	tree-sitter-ruby tree-sitter-elixir tree-sitter-go

SO_TARGETS := $(patsubst %,$(GRAMMARS_DIR)/%.so,$(GRAMMAR_NAMES))

# ==============================================================================
# Main targets
# ==============================================================================

.PHONY: grammars clean-grammars list-grammars help

grammars: $(SO_TARGETS)
	@echo ""
	@echo "All tree-sitter grammars compiled and installed to $(GRAMMARS_DIR)/"
	@ls -lh $(GRAMMARS_DIR)/*.so

help:
	@echo "Docker / Training Targets:"
	@echo "  make train-image-rocm - Build ROCm GPU training image"
	@echo "  make train-image-cuda - Build CUDA GPU training image"
	@echo "  make app-image        - Build application image (CPU)"
	@echo "  make login            - Log in to GHCR (set CR_PAT env var)"
	@echo "  make push-images      - Push all images to GHCR"
	@echo "  make deploy           - Deploy full stack to Docker Swarm"
	@echo "  make stack-down       - Remove the stack from Docker Swarm"
	@echo "  make train-rocm       - Trigger ROCm training job"
	@echo "  make train-cuda       - Trigger CUDA training job"
	@echo "  make clean-volumes    - Remove all build cache volumes"
	@echo ""
	@echo "Tree-sitter Grammar Targets:"
	@echo "  make grammars         - Build all 9 grammars"
	@echo "  make grammar-elixir   - Build just Elixir grammar"
	@echo "  make grammar-python   - Build just Python grammar"
	@echo "  make grammar-c        - Build just C grammar"
	@echo "  make grammar-cpp      - Build just C++ grammar"
	@echo "  make grammar-java     - Build just Java grammar"
	@echo "  make grammar-csharp   - Build just C# grammar"
	@echo "  make grammar-php      - Build just PHP grammar"
	@echo "  make grammar-ruby     - Build just Ruby grammar"
	@echo "  make grammar-go       - Build just Go grammar"
	@echo "  make list-grammars    - Show status of all grammars"
	@echo "  make clean-grammars   - Remove vendor sources and compiled libs"

clean-grammars:
	rm -rf $(VENDOR_DIR)
	rm -f $(GRAMMARS_DIR)/*.so
	@echo "Cleaned tree-sitter grammars"

list-grammars:
	@echo "Grammar status ($(GRAMMARS_DIR)):"
	@for name in $(GRAMMAR_NAMES); do \
		if [ -f "$(GRAMMARS_DIR)/$$name.so" ]; then \
			size=$$(du -h "$(GRAMMARS_DIR)/$$name.so" | cut -f1); \
			echo "  ✓ $$name.so ($$size)"; \
		else \
			echo "  ✗ $$name.so (not built)"; \
		fi; \
	done

# ==============================================================================
# Convenience per-language targets
# ==============================================================================

.PHONY: grammar-c grammar-cpp grammar-java grammar-csharp grammar-php \
	grammar-python grammar-ruby grammar-elixir grammar-go

grammar-c:       $(GRAMMARS_DIR)/tree-sitter-c.so
grammar-cpp:     $(GRAMMARS_DIR)/tree-sitter-cpp.so
grammar-java:    $(GRAMMARS_DIR)/tree-sitter-java.so
grammar-csharp:  $(GRAMMARS_DIR)/tree-sitter-c-sharp.so
grammar-php:     $(GRAMMARS_DIR)/tree-sitter-php.so
grammar-python:  $(GRAMMARS_DIR)/tree-sitter-python.so
grammar-ruby:    $(GRAMMARS_DIR)/tree-sitter-ruby.so
grammar-elixir:  $(GRAMMARS_DIR)/tree-sitter-elixir.so
grammar-go:      $(GRAMMARS_DIR)/tree-sitter-go.so

# ==============================================================================
# Clone helper
# ==============================================================================

# url_for: extract URL for a grammar name from GRAMMAR_REPOS
url_for = $(patsubst $(1)=%,%,$(filter $(1)=%,$(GRAMMAR_REPOS)))

define clone_repo
	@mkdir -p $(VENDOR_DIR)
	@if [ ! -d "$(VENDOR_DIR)/$(1)" ]; then \
		echo "Cloning $(1)..."; \
		git clone --depth 1 --quiet $(call url_for,$(1)) $(VENDOR_DIR)/$(1); \
	fi
endef

# ==============================================================================
# Compile helper: finds src/parser.c and any scanner files, compiles to .so
#
# Tree-sitter grammars have:
#   src/parser.c            - always present (generated)
#   src/scanner.c           - optional C scanner
#   src/scanner.cc          - optional C++ scanner (older grammars)
#
# PHP is special: sources are in php/src/ instead of src/
# ==============================================================================

define compile_grammar
	@mkdir -p $(GRAMMARS_DIR)
	@SRC="$(VENDOR_DIR)/$(1)/src"; \
	if [ "$(1)" = "tree-sitter-php" ] && [ -d "$(VENDOR_DIR)/$(1)/php/src" ]; then \
		SRC="$(VENDOR_DIR)/$(1)/php/src"; \
	fi; \
	echo "Compiling $(1) from $$SRC..."; \
	SOURCES="$$SRC/parser.c"; \
	if [ -f "$$SRC/scanner.c" ]; then \
		SOURCES="$$SOURCES $$SRC/scanner.c"; \
	fi; \
	if [ -f "$$SRC/scanner.cc" ]; then \
		$(CXX) $(CXXFLAGS) -I $$SRC $$SOURCES $$SRC/scanner.cc -o $(GRAMMARS_DIR)/$(1).so; \
	else \
		$(CC) $(CFLAGS) -I $$SRC $$SOURCES -o $(GRAMMARS_DIR)/$(1).so; \
	fi; \
	echo "  -> $(GRAMMARS_DIR)/$(1).so"
endef

# ==============================================================================
# Per-grammar build rules
# ==============================================================================

$(GRAMMARS_DIR)/tree-sitter-c.so:
	$(call clone_repo,tree-sitter-c)
	$(call compile_grammar,tree-sitter-c)

$(GRAMMARS_DIR)/tree-sitter-cpp.so:
	$(call clone_repo,tree-sitter-cpp)
	$(call compile_grammar,tree-sitter-cpp)

$(GRAMMARS_DIR)/tree-sitter-java.so:
	$(call clone_repo,tree-sitter-java)
	$(call compile_grammar,tree-sitter-java)

$(GRAMMARS_DIR)/tree-sitter-c-sharp.so:
	$(call clone_repo,tree-sitter-c-sharp)
	$(call compile_grammar,tree-sitter-c-sharp)

$(GRAMMARS_DIR)/tree-sitter-php.so:
	$(call clone_repo,tree-sitter-php)
	$(call compile_grammar,tree-sitter-php)

$(GRAMMARS_DIR)/tree-sitter-python.so:
	$(call clone_repo,tree-sitter-python)
	$(call compile_grammar,tree-sitter-python)

$(GRAMMARS_DIR)/tree-sitter-ruby.so:
	$(call clone_repo,tree-sitter-ruby)
	$(call compile_grammar,tree-sitter-ruby)

$(GRAMMARS_DIR)/tree-sitter-elixir.so:
	$(call clone_repo,tree-sitter-elixir)
	$(call compile_grammar,tree-sitter-elixir)

$(GRAMMARS_DIR)/tree-sitter-go.so:
	$(call clone_repo,tree-sitter-go)
	$(call compile_grammar,tree-sitter-go)
