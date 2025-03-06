PROJECT_NAME := geoflex
VENV_BASE_PATH ?= ~/.testing_envs
VENV_PATH ?= $(VENV_BASE_PATH)/$(PROJECT_NAME)_venv

.PHONY: venv test clean
venv:
				@echo "Creating virtual environment at $(VENV_PATH)..."
				mkdir -p $(VENV_BASE_PATH)
				python3 -m venv $(VENV_PATH)
				$(VENV_PATH)/bin/pip install -e '.[dev]'
test: $(VENV_PATH)
				@echo "Running tests..."
				$(VENV_PATH)/bin/pytest -p no:cacheprovider
clean:
				@echo "Cleaning up virtual environment at $(VENV_PATH)..."
				rm -rf $(VENV_PATH) ../../$(PROJECT_NAME).egg-info
				@echo "Virtual environment removed."
# This rule ensures that if you call make test without the venv existing, it will be created
$(VENV_PATH):
				@$(MAKE) venv VENV_PATH=$@