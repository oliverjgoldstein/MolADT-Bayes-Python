SYSTEM_PYTHON := $(if $(wildcard /opt/homebrew/bin/python3),/opt/homebrew/bin/python3,python3)
PYTHON_CMD := $(if $(wildcard .venv/bin/python),./.venv/bin/python,$(SYSTEM_PYTHON))

INFERENCE_PRESET ?= default
QM9_LIMIT ?= 2000
ZINC_DATASET_SIZE ?= 250K
ZINC_DATASET_DIMENSION ?= 2D
ZINC_LIMIT ?=
EXAMPLE ?= ferrocene
MODELS ?= bayes_linear_student_t,bayes_hierarchical_shrinkage
BENCHMARK_LOG ?= results/benchmark.out
BENCHMARK_PID ?= results/benchmark.pid

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
RUNTIME_HINT := often 3-6 hours on an M1 Pro, with QM9_LIMIT=5000 and the full ZINC 250K timing pass
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
ZINC_LIMIT_BENCHMARK_ARG := $(if $(ZINC_LIMIT),--zinc-limit $(ZINC_LIMIT),)
ZINC_LIMIT_TIMING_ARG := $(if $(ZINC_LIMIT),--limit $(ZINC_LIMIT),)

.PHONY: help python-setup python-cmdstan-install python-test python-typecheck python-parse python-parse-smiles python-to-smiles python-pretty-example python-benchmark-smoke python-benchmark-qm9 python-benchmark-zinc benchmark benchmark-bg

help:
	@printf "%s\n" \
	"Python repo targets:" \
	"  make python-setup           Create .venv and install the package plus benchmark deps" \
	"  make python-cmdstan-install Install CmdStan into .cmdstan" \
	"  make python-test            Run the pytest suite" \
	"  make python-typecheck       Run mypy on the package" \
	"  make python-parse           Parse molecules/benzene.sdf" \
	"  make python-parse-smiles    Parse c1ccccc1" \
	"  make python-to-smiles       Render molecules/benzene.sdf to SMILES" \
	"  make python-pretty-example  Render EXAMPLE=ferrocene or EXAMPLE=diborane" \
	"  make benchmark              Run FreeSolv, QM9, and ZINC benchmark flows" \
	"  make benchmark-bg           Run the benchmark in the background" \
	"" \
	"Current inference configuration:" \
	"  preset=$(INFERENCE_PRESET)" \
	"  methods=$(METHODS)" \
	"  models=$(MODELS)" \
	"  sample_chains=$(SAMPLE_CHAINS) sample_warmup=$(SAMPLE_WARMUP) sample_draws=$(SAMPLE_DRAWS)" \
	"  approximation_draws=$(APPROXIMATION_DRAWS) pathfinder_paths=$(PATHFINDER_PATHS)" \
	"  rough runtime: $(RUNTIME_HINT)"

python-setup:
	for attempt in 1 2 3; do \
		[ ! -e .venv ] && break; \
		find .venv -name '.DS_Store' -delete 2>/dev/null || true; \
		rm -rf .venv 2>/dev/null || true; \
		find .venv -name '.DS_Store' -delete 2>/dev/null || true; \
		rmdir .venv/lib/python3.14 .venv/lib/python3.13 .venv/lib/python3.12 .venv/lib/python3.11 .venv/lib/python3.10 .venv/lib .venv 2>/dev/null || true; \
		sleep 1; \
	done && \
	[ ! -e .venv ] && \
	$(SYSTEM_PYTHON) -m venv .venv && \
	.venv/bin/python -m pip install -U pip setuptools wheel && \
	.venv/bin/python -m pip install -U -e ".[dev]"

python-cmdstan-install:
	$(PYTHON_CMD) -c "import cmdstanpy; cmdstanpy.install_cmdstan(dir='.cmdstan')"

python-test:
	$(PYTHON_CMD) -m pytest -q

python-typecheck:
	$(PYTHON_CMD) -m mypy moladt

python-parse:
	$(PYTHON_CMD) -m moladt.cli parse molecules/benzene.sdf

python-parse-smiles:
	$(PYTHON_CMD) -m moladt.cli parse-smiles "c1ccccc1"

python-to-smiles:
	$(PYTHON_CMD) -m moladt.cli to-smiles molecules/benzene.sdf

python-pretty-example:
	$(PYTHON_CMD) -m moladt.cli pretty-example $(EXAMPLE)

python-benchmark-smoke:
	$(PYTHON_CMD) -m scripts.run_all smoke-test $(BENCHMARK_ARGS)

python-benchmark-qm9:
	$(PYTHON_CMD) -m scripts.run_all qm9 --limit $(QM9_LIMIT) $(BENCHMARK_ARGS)

python-benchmark-zinc:
	$(PYTHON_CMD) -m scripts.run_all zinc-timing --dataset-size $(ZINC_DATASET_SIZE) --dataset-dimension $(ZINC_DATASET_DIMENSION) $(ZINC_LIMIT_TIMING_ARG)

benchmark:
	$(PYTHON_CMD) -m scripts.run_all benchmark --qm9-limit $(QM9_LIMIT) --zinc-dataset-size $(ZINC_DATASET_SIZE) --zinc-dataset-dimension $(ZINC_DATASET_DIMENSION) $(ZINC_LIMIT_BENCHMARK_ARG) $(BENCHMARK_ARGS)

benchmark-bg:
	@mkdir -p $(dir $(BENCHMARK_LOG))
	@nohup $(MAKE) --no-print-directory benchmark > $(BENCHMARK_LOG) 2>&1 & echo $$! > $(BENCHMARK_PID)
	@printf "%s\n" \
	"Started benchmark in the background." \
	"  pid file: $(BENCHMARK_PID)" \
	"  log file: $(BENCHMARK_LOG)" \
	"  report: results/model_report.md" \
	"  coefficients: results/model_coefficients.csv"
