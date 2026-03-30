from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_PATH = WORKSPACE_ROOT / "scripts" / "showcase.py"


def _load_showcase_module():
    spec = importlib.util.spec_from_file_location("workspace_showcase", SHOWCASE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_copy_python_result_files_mirrors_csv_first_layout(tmp_path) -> None:
    showcase = _load_showcase_module()
    source_root = tmp_path / "results" / "run_20260330_170000"
    artifacts_dir = tmp_path / "showcase-artifacts"
    copied_results_dir = artifacts_dir / "python-results"

    (copied_results_dir / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (copied_results_dir / "summary.md").write_text("stale\n", encoding="utf-8")

    for relative_name in (
        "results.csv",
        "rmse_train_test_vs_literature.svg",
        "details/predictive_metrics.csv",
        "details/model_coefficients.csv",
    ):
        target = source_root / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("payload\n", encoding="utf-8")

    showcase.ARTIFACTS_DIR = artifacts_dir

    copied_paths = showcase._copy_python_result_files(source_root)

    assert copied_results_dir.exists()
    assert not (copied_results_dir / "summary.md").exists()
    assert (copied_results_dir / "results.csv").exists()
    assert (copied_results_dir / "details" / "predictive_metrics.csv").exists()
    assert (copied_results_dir / "details" / "model_coefficients.csv").exists()
    assert [path.relative_to(artifacts_dir).as_posix() for path in copied_paths] == [
        "python-results/results.csv",
        "python-results/rmse_train_test_vs_literature.svg",
        "python-results/details/predictive_metrics.csv",
        "python-results/details/model_coefficients.csv",
    ]


def test_build_results_snapshot_formats_predictive_and_timing_rows(tmp_path) -> None:
    showcase = _load_showcase_module()
    results_csv = tmp_path / "results.csv"
    results_csv.write_text(
        "row_type,task,model,method,train_rmse,test_rmse,test_minus_train_rmse,stage,molecule_count,success_count,failure_count,molecules_per_second\n"
        "predictive_summary,freesolv / smiles,m1,optimize,1.2,1.4,0.2,,,,,\n"
        "timing_stage,,,,,,,raw_file_read,100,100,0,200.0\n",
        encoding="utf-8",
    )

    snapshot = showcase._build_results_snapshot(results_csv)

    assert "### Predictive Summary" in snapshot
    assert "| freesolv / smiles | m1 | optimize | 1.2 | 1.4 | 0.2 |" in snapshot
    assert "### Timing Summary" in snapshot
    assert "| raw_file_read | 100 | 100 | 0 | 200.0 |" in snapshot
