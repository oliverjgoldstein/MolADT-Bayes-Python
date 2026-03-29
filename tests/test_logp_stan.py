from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from moladt.inference.logp import read_logp_observations, read_named_molecules, run_logp_regression


PROJECT_ROOT = Path(__file__).resolve().parent.parent


pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("stan") is None, reason="stan is not installed"),
    pytest.mark.skipif(sys.version_info >= (3, 13), reason="PyStan smoke test is scoped to Python 3.11/3.12"),
]
os.environ.setdefault("HTTPSTAN_DEBUG", "1")


def test_stan_smoke_sample() -> None:
    training = read_logp_observations(PROJECT_ROOT / "logp" / "DB1.sdf", limit=8)
    water = read_named_molecules(PROJECT_ROOT / "molecules" / "water.sdf")
    result = run_logp_regression(
        training,
        water,
        num_chains=1,
        num_samples=20,
        num_warmup=20,
        seed=1,
    )
    assert result.posterior_draw_count > 0
    assert len(result.test_predictions) == 1
