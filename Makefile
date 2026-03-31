SYSTEM_PYTHON ?= $(strip $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null))
OS_NAME ?= $(strip $(shell uname -s 2>/dev/null))
XCRUN ?= $(strip $(shell command -v xcrun 2>/dev/null))
TESTED_PYTHON := 3.14.3
TESTED_CMDSTANPY := 1.3.0
TESTED_CMDSTAN := 2.38.0
TESTED_RDKIT := 2025.9.6
VENV_PYTHON_UNIX := .venv/bin/python
VENV_PYTHON_WIN := .venv/Scripts/python.exe
VENV_PYTHON_WIN_ALT := .venv/Scripts/python
PYTHON_CMD := $(if $(wildcard $(VENV_PYTHON_UNIX)),./$(VENV_PYTHON_UNIX),$(if $(wildcard $(VENV_PYTHON_WIN)),./$(VENV_PYTHON_WIN),$(if $(wildcard $(VENV_PYTHON_WIN_ALT)),./$(VENV_PYTHON_WIN_ALT),$(SYSTEM_PYTHON))))
BASH := $(strip $(shell command -v bash 2>/dev/null || printf "%s" /bin/bash))
APT_GET ?= apt-get
SUDO ?= sudo
AUTO_INSTALL_VENV ?= 1
AUTO_FIX_PROMPT ?= 1
AUTO_APPROVE_FIXES ?= 0

INFERENCE_PRESET ?= default
QM9_LIMIT ?= 2000
QM9_SPLIT_MODE ?= subset
ZINC_DATASET_SIZE ?= 250K
ZINC_DATASET_DIMENSION ?= 2D
ZINC_LIMIT ?=
INCLUDE_MOLADT ?= 0
BENCHMARK_VERBOSE ?= 1
EXAMPLE ?= ferrocene
MODELS ?= bayes_linear_student_t,bayes_hierarchical_shrinkage
PYTHON_EXTRAS ?= dev,ml,geom
RESULTS_SUBDIR ?=
RUN_TIMESTAMP ?= $(shell date +%Y%m%d_%H%M%S)

ifeq ($(INFERENCE_PRESET),quick)
METHODS_DEFAULT := sample,pathfinder,optimize
SAMPLE_CHAINS_DEFAULT := 1
SAMPLE_WARMUP_DEFAULT := 50
SAMPLE_DRAWS_DEFAULT := 50
APPROXIMATION_DRAWS_DEFAULT := 100
VARIATIONAL_ITERATIONS_DEFAULT := 3000
OPTIMIZE_ITERATIONS_DEFAULT := 1500
PATHFINDER_PATHS_DEFAULT := 2
PREDICTIVE_DRAWS_DEFAULT := 250
RUNTIME_HINT := usually 5-15 minutes after CmdStan is built, with QM9_LIMIT=500 and ZINC_LIMIT=10000
else ifeq ($(INFERENCE_PRESET),paper)
METHODS_DEFAULT := sample,variational,pathfinder,optimize,laplace
SAMPLE_CHAINS_DEFAULT := 4
SAMPLE_WARMUP_DEFAULT := 1500
SAMPLE_DRAWS_DEFAULT := 1500
APPROXIMATION_DRAWS_DEFAULT := 2000
VARIATIONAL_ITERATIONS_DEFAULT := 15000
OPTIMIZE_ITERATIONS_DEFAULT := 5000
PATHFINDER_PATHS_DEFAULT := 8
PREDICTIVE_DRAWS_DEFAULT := 2000
RUNTIME_HINT := often several hours on an M1 Pro, with the paper-sized QM9 split (110462/10000/10000) and the full ZINC 250K timing pass
else
METHODS_DEFAULT := sample,variational,pathfinder,optimize,laplace
SAMPLE_CHAINS_DEFAULT := 2
SAMPLE_WARMUP_DEFAULT := 200
SAMPLE_DRAWS_DEFAULT := 200
APPROXIMATION_DRAWS_DEFAULT := 500
VARIATIONAL_ITERATIONS_DEFAULT := 5000
OPTIMIZE_ITERATIONS_DEFAULT := 2000
PATHFINDER_PATHS_DEFAULT := 4
PREDICTIVE_DRAWS_DEFAULT := 500
RUNTIME_HINT := usually 15-45 minutes after CmdStan is built, with QM9_LIMIT=2000 and the full ZINC 250K timing pass
endif

ifeq ($(INFERENCE_PRESET),paper)
ifeq ($(strip $(RESULTS_SUBDIR)),)
RESULTS_SUBDIR := paper/run_$(RUN_TIMESTAMP)
endif
else ifeq ($(strip $(RESULTS_SUBDIR)),)
RESULTS_SUBDIR := run_$(RUN_TIMESTAMP)
endif

RESULTS_ROOT := $(if $(RESULTS_SUBDIR),results/$(RESULTS_SUBDIR),results)
RESULTS_ENV := MOLADT_RESULTS_DIR=$(RESULTS_ROOT)
BENCHMARK_LOG ?= $(RESULTS_ROOT)/benchmark.out

METHODS ?= $(METHODS_DEFAULT)
SAMPLE_CHAINS ?= $(SAMPLE_CHAINS_DEFAULT)
SAMPLE_WARMUP ?= $(SAMPLE_WARMUP_DEFAULT)
SAMPLE_DRAWS ?= $(SAMPLE_DRAWS_DEFAULT)
APPROXIMATION_DRAWS ?= $(APPROXIMATION_DRAWS_DEFAULT)
VARIATIONAL_ITERATIONS ?= $(VARIATIONAL_ITERATIONS_DEFAULT)
OPTIMIZE_ITERATIONS ?= $(OPTIMIZE_ITERATIONS_DEFAULT)
PATHFINDER_PATHS ?= $(PATHFINDER_PATHS_DEFAULT)
PREDICTIVE_DRAWS ?= $(PREDICTIVE_DRAWS_DEFAULT)

BENCHMARK_ARGS := --methods $(METHODS) --models $(MODELS) --sample-chains $(SAMPLE_CHAINS) --sample-warmup $(SAMPLE_WARMUP) --sample-draws $(SAMPLE_DRAWS) --approximation-draws $(APPROXIMATION_DRAWS) --variational-iterations $(VARIATIONAL_ITERATIONS) --optimize-iterations $(OPTIMIZE_ITERATIONS) --pathfinder-paths $(PATHFINDER_PATHS) --predictive-draws $(PREDICTIVE_DRAWS)
QM9_LIMIT_QM9_ARG := $(if $(QM9_LIMIT),--limit $(QM9_LIMIT),)
QM9_LIMIT_BENCHMARK_ARG := $(if $(QM9_LIMIT),--qm9-limit $(QM9_LIMIT),)
ZINC_LIMIT_BENCHMARK_ARG := $(if $(ZINC_LIMIT),--zinc-limit $(ZINC_LIMIT),)
ZINC_LIMIT_TIMING_ARG := $(if $(ZINC_LIMIT),--limit $(ZINC_LIMIT),)
INCLUDE_MOLADT_ARG := $(if $(filter 1 true yes TRUE YES,$(INCLUDE_MOLADT)),--include-moladt,)
VERBOSE_ARG := $(if $(filter 1 true yes TRUE YES,$(BENCHMARK_VERBOSE)),--verbose,)

ifeq ($(OS_NAME),Darwin)
ifneq ($(XCRUN),)
DARWIN_CLANG := $(strip $(shell $(XCRUN) --find clang 2>/dev/null))
DARWIN_CLANGXX := $(strip $(shell $(XCRUN) --find clang++ 2>/dev/null))
DARWIN_SDKROOT := $(strip $(shell $(XCRUN) --show-sdk-path 2>/dev/null))
endif
endif

TOOLCHAIN_ENV := $(if $(DARWIN_SDKROOT),env CC="$(DARWIN_CLANG)" CXX="$(DARWIN_CLANGXX)" SDKROOT="$(DARWIN_SDKROOT)" CFLAGS="$${CFLAGS:+$$CFLAGS }-isysroot $(DARWIN_SDKROOT)" CXXFLAGS="$${CXXFLAGS:+$$CXXFLAGS }-isysroot $(DARWIN_SDKROOT)",)

MODEL_RESULTS_SUBDIR := $(if $(filter paper,$(INFERENCE_PRESET)),models/paper/run_$(RUN_TIMESTAMP),models/run_$(RUN_TIMESTAMP))

.PHONY: help python-setup python-cmdstan-install python-test python-typecheck python-activate python-parse python-parse-smiles python-to-smiles python-pretty-example python-benchmark-smoke python-benchmark-qm9 python-benchmark-zinc benchmark benchmark-small benchmark-paper benchmark-bg model

help:
	@printf "%s\n" \
	"Python repo targets:" \
	"  make python-setup           Create .venv and install the package plus model deps" \
	"  make python-cmdstan-install Install CmdStan into .cmdstan" \
	"  make python-test            Run the pytest suite" \
	"  make python-typecheck       Run mypy on the package" \
	"  make python-activate        Print the command that activates the local venv" \
	"  make python-parse           Parse molecules/benzene.sdf" \
	"  make python-parse-smiles    Parse c1ccccc1" \
	"  make python-to-smiles       Render molecules/benzene.sdf to SMILES" \
		"  make python-pretty-example  Render EXAMPLE=ferrocene or EXAMPLE=diborane" \
		"  make benchmark              Run FreeSolv, QM9, and ZINC benchmark flows" \
		"  make benchmark-small        Run the default 2000-row QM9 subset benchmark with MolADT timing enabled" \
		"  make benchmark-paper        Run the paper-sized QM9 split (110462/10000/10000) with MolADT timing enabled" \
		"  make benchmark-bg           Run the benchmark in the foreground and mirror output to the active results directory" \
		"  make model                  Run the predictive model suite and write per-model folders" \
		"  full long run: make benchmark-paper" \
		"  quieter run: BENCHMARK_VERBOSE=0 make benchmark" \
		"" \
		"Current inference configuration:" \
		"  preset=$(INFERENCE_PRESET)" \
		"  results_root=$(RESULTS_ROOT)" \
		"  qm9_split_mode=$(QM9_SPLIT_MODE)" \
		"  qm9_limit=$(if $(QM9_LIMIT),$(QM9_LIMIT),full-local-download)" \
	"  include_moladt=$(INCLUDE_MOLADT)" \
	"  benchmark_verbose=$(BENCHMARK_VERBOSE)" \
	"  python_extras=$(PYTHON_EXTRAS)" \
	"  toolchain_env=$(if $(DARWIN_SDKROOT),apple-xcrun,default)" \
	"  methods=$(METHODS)" \
	"  models=$(MODELS)" \
	"  sample_chains=$(SAMPLE_CHAINS) sample_warmup=$(SAMPLE_WARMUP) sample_draws=$(SAMPLE_DRAWS)" \
	"  approximation_draws=$(APPROXIMATION_DRAWS) pathfinder_paths=$(PATHFINDER_PATHS)" \
	"  tested Python=$(TESTED_PYTHON) cmdstanpy=$(TESTED_CMDSTANPY) CmdStan=$(TESTED_CMDSTAN) RDKit=$(TESTED_RDKIT)" \
	"  rough runtime: $(RUNTIME_HINT)"

python-setup:
	@set -e; \
	system_python="$(SYSTEM_PYTHON)"; \
	auto_install_venv="$(AUTO_INSTALL_VENV)"; \
	prompt_fixes="$(AUTO_FIX_PROMPT)"; \
	auto_approve_fixes="$(AUTO_APPROVE_FIXES)"; \
	apt_get_cmd="$(APT_GET)"; \
	sudo_cmd="$(SUDO)"; \
	venv_error_log=".venv-setup.log"; \
	confirm_fix() { \
		prompt_text="$$1"; \
		if [ "$$auto_approve_fixes" = "1" ]; then \
			printf "%s\n" "$$prompt_text" "Auto-approving this repair."; \
			return 0; \
		fi; \
		if [ "$$prompt_fixes" = "0" ]; then \
			return 1; \
		fi; \
		printf "%s" "$$prompt_text"; \
		IFS= read -r response || response=""; \
		printf "%s\n" ""; \
		case "$$response" in \
			[Yy]|[Yy][Ee][Ss]) return 0 ;; \
			*) return 1 ;; \
		esac; \
	}; \
	if [ -z "$$system_python" ]; then \
		printf "%s\n" \
		"" \
		"Could not find a Python interpreter." \
		"Install Python 3.11+ and ensure \`python3\` or \`python\` is on PATH, then rerun:" \
		"  make python-setup"; \
		exit 1; \
	fi; \
	if ! "$$system_python" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"; then \
		python_version="$$( "$$system_python" -V 2>&1 || printf "%s" "unknown version" )"; \
		printf "%s\n" \
		"" \
		"MolADT-Bayes-Python requires Python 3.11+ for setup." \
		"Found: $$python_version" \
		"Install Python 3.11+ and rerun:" \
		"  make python-setup"; \
		exit 1; \
	fi; \
	"$$system_python" -c "from pathlib import Path; import shutil; path = Path('.venv'); shutil.rmtree(path, ignore_errors=True) if path.exists() else None"; \
	if [ -e .venv ]; then \
		printf "%s\n" \
		"" \
		"Could not remove the existing .venv directory." \
		"Delete it manually and rerun:" \
		"  rm -rf .venv" \
		"  make python-setup"; \
		exit 1; \
	fi; \
	rm -f "$$venv_error_log"; \
	create_venv() { "$$system_python" -m venv .venv 2>"$$venv_error_log"; }; \
	if ! create_venv; then \
		auto_fixed=0; \
		is_root=0; \
		if command -v id >/dev/null 2>&1 && [ "$$(id -u)" = "0" ]; then \
			is_root=1; \
		fi; \
		pkg_prefix=""; \
		if [ "$$is_root" != "1" ] && { [ -x "$$sudo_cmd" ] || command -v "$$sudo_cmd" >/dev/null 2>&1; }; then \
			pkg_prefix="$$sudo_cmd"; \
		fi; \
		can_use_apt=0; \
		if [ "$$auto_install_venv" != "0" ] && [ -s "$$venv_error_log" ] && grep -Eiq 'ensurepip is not available|No module named ensurepip' "$$venv_error_log" && { [ -x "$$apt_get_cmd" ] || command -v "$$apt_get_cmd" >/dev/null 2>&1; } && { [ "$$is_root" = "1" ] || [ -n "$$pkg_prefix" ]; }; then \
			can_use_apt=1; \
		fi; \
		run_apt() { \
			if [ -n "$$pkg_prefix" ]; then \
				"$$pkg_prefix" "$$apt_get_cmd" "$$@"; \
			else \
				"$$apt_get_cmd" "$$@"; \
			fi; \
		}; \
		if [ "$$can_use_apt" = "1" ] && confirm_fix "Detected missing ensurepip support while creating .venv. Install the Linux venv package now? [y/N] "; then \
			python_short_version="$$( "$$system_python" -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))' )"; \
			printf "%s\n" \
				"" \
				"Detected missing ensurepip support while creating .venv." \
				"Attempting automatic installation of Linux venv packages."; \
			if run_apt update; then \
				for package in "python3-venv" "python$${python_short_version}-venv"; do \
					printf "%s\n" "Trying package: $$package"; \
					if run_apt install -y "$$package"; then \
						"$$system_python" -c "from pathlib import Path; import shutil; path = Path('.venv'); shutil.rmtree(path, ignore_errors=True) if path.exists() else None"; \
						if create_venv; then \
							auto_fixed=1; \
							break; \
						fi; \
					fi; \
				done; \
			fi; \
		fi; \
		if [ "$$auto_fixed" != "1" ]; then \
			printf "%s\n" \
				"" \
				"Python could not create .venv." \
				"If you are on WSL, Debian, or Ubuntu, install the venv package for your Linux Python first:" \
				"  sudo apt update" \
				"  sudo apt install -y python3-venv" \
				"" \
				"If your distro uses a versioned package name instead, install the one that matches \`python3 --version\`," \
				"for example \`python3.10-venv\` or \`python3.12-venv\`." \
				"" \
				"Then rerun:" \
				"  make python-setup"; \
			if [ -s "$$venv_error_log" ]; then \
				printf "%s\n" "" "Original venv error:"; \
				sed -n '1,20p' "$$venv_error_log"; \
			fi; \
			exit 1; \
		fi; \
	fi; \
	rm -f "$$venv_error_log"; \
	venv_python=""; \
	if [ -f "$(VENV_PYTHON_UNIX)" ]; then \
		venv_python="$(VENV_PYTHON_UNIX)"; \
	elif [ -f "$(VENV_PYTHON_WIN)" ]; then \
		venv_python="$(VENV_PYTHON_WIN)"; \
	elif [ -f "$(VENV_PYTHON_WIN_ALT)" ]; then \
		venv_python="$(VENV_PYTHON_WIN_ALT)"; \
	else \
		printf "%s\n" \
		"" \
		"Created .venv, but could not find its Python executable." \
		"Expected one of:" \
		"  $(VENV_PYTHON_UNIX)" \
		"  $(VENV_PYTHON_WIN)"; \
		exit 1; \
	fi; \
	"$$venv_python" -m pip install -U pip setuptools wheel; \
	"$$venv_python" -m pip install -U -e ".[${PYTHON_EXTRAS}]"

python-cmdstan-install:
	$(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.install_cmdstan

python-test:
	@start=$$(date +%s); \
	printf "%s\n" "Starting Python tests at $$(date '+%H:%M:%S'). Typical runtime: about 5-15 seconds."; \
	$(PYTHON_CMD) -m pytest -q; \
	status=$$?; \
	end=$$(date +%s); \
	printf "%s\n" "Finished Python tests in $$((end-start)) seconds."; \
	exit $$status

python-typecheck:
	@start=$$(date +%s); \
	printf "%s\n" "Starting Python typecheck at $$(date '+%H:%M:%S'). Typical runtime: about 5-15 seconds."; \
	$(PYTHON_CMD) -m mypy moladt; \
	status=$$?; \
	end=$$(date +%s); \
	printf "%s\n" "Finished Python typecheck in $$((end-start)) seconds."; \
	exit $$status

python-activate:
	@if [ -f .venv/bin/activate ]; then \
		printf "%s\n" "Run this in your shell:" "  source .venv/bin/activate"; \
	elif [ -f .venv/Scripts/activate ]; then \
		printf "%s\n" "Run this in your shell:" "  source .venv/Scripts/activate"; \
	else \
		printf "%s\n" "Run this in your shell after \`make python-setup\`:" "  source .venv/bin/activate" "  or, if your environment created a Windows-style venv:" "  source .venv/Scripts/activate"; \
	fi

python-parse:
	$(PYTHON_CMD) -m moladt.cli parse molecules/benzene.sdf

python-parse-smiles:
	$(PYTHON_CMD) -m moladt.cli parse-smiles "c1ccccc1"

python-to-smiles:
	$(PYTHON_CMD) -m moladt.cli to-smiles molecules/benzene.sdf

python-pretty-example:
	$(PYTHON_CMD) -m moladt.cli pretty-example $(EXAMPLE)

python-benchmark-smoke:
	$(RESULTS_ENV) $(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.run_all smoke-test $(VERBOSE_ARG) $(BENCHMARK_ARGS)

python-benchmark-qm9:
	$(RESULTS_ENV) $(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.run_all qm9 $(QM9_LIMIT_QM9_ARG) --split-mode $(QM9_SPLIT_MODE) $(VERBOSE_ARG) $(BENCHMARK_ARGS)

python-benchmark-zinc:
	$(RESULTS_ENV) $(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.run_all zinc-timing --dataset-size $(ZINC_DATASET_SIZE) --dataset-dimension $(ZINC_DATASET_DIMENSION) $(ZINC_LIMIT_TIMING_ARG) $(INCLUDE_MOLADT_ARG) $(VERBOSE_ARG)

benchmark:
	$(RESULTS_ENV) $(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.run_all benchmark $(QM9_LIMIT_BENCHMARK_ARG) --qm9-split-mode $(QM9_SPLIT_MODE) --zinc-dataset-size $(ZINC_DATASET_SIZE) --zinc-dataset-dimension $(ZINC_DATASET_DIMENSION) $(ZINC_LIMIT_BENCHMARK_ARG) $(INCLUDE_MOLADT_ARG) $(VERBOSE_ARG) $(BENCHMARK_ARGS)

benchmark-small:
	@$(MAKE) --no-print-directory benchmark QM9_LIMIT=2000 QM9_SPLIT_MODE=subset INCLUDE_MOLADT=1

benchmark-paper:
	@$(MAKE) --no-print-directory benchmark INFERENCE_PRESET=paper QM9_LIMIT= QM9_SPLIT_MODE=paper INCLUDE_MOLADT=1

benchmark-bg:
	@mkdir -p $(dir $(BENCHMARK_LOG))
	@printf "%s\n" \
	"Running benchmark in the foreground." \
	"  output is mirrored to: $(BENCHMARK_LOG)" \
	"  summary csv: $(RESULTS_ROOT)/results.csv" \
	"  details dir: $(RESULTS_ROOT)/details/"
	@$(BASH) -o pipefail -c '$(MAKE) --no-print-directory benchmark 2>&1 | tee "$(BENCHMARK_LOG)"'

model:
	MOLADT_RESULTS_DIR=results/$(MODEL_RESULTS_SUBDIR) $(TOOLCHAIN_ENV) $(PYTHON_CMD) -m scripts.run_all models $(QM9_LIMIT_BENCHMARK_ARG) --qm9-split-mode $(QM9_SPLIT_MODE) $(if $(filter paper,$(INFERENCE_PRESET)),--paper-mode,) $(VERBOSE_ARG) $(BENCHMARK_ARGS)
