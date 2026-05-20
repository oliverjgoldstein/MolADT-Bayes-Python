from __future__ import annotations

import pytest

from benchmarking.freesolv_small_feature_gp import (
    WL_MODEL_KEY,
    _tanimoto,
    _token_views,
    freesolv_model_keys_from_names,
)
from moladt.examples.sample_molecules import water


def test_gp_wl_model_aliases_are_accepted_for_freesolv_runs() -> None:
    assert freesolv_model_keys_from_names(("GP_WL", "moladt_full30_rbf_gp")) == (
        WL_MODEL_KEY,
        "full_moladt",
    )


def test_gp_wl_tokens_include_orbital_aware_atom_labels() -> None:
    view = _token_views(water)

    assert any(
        name.startswith("wl0:O:neutral:sh2:orb5:e8:1s2.2s2.2p211")
        for name in view.wl_tokens
    )
    assert any(
        name.startswith("atom_shell:O:neutral:sh2:orb5:e8:1s2.2s2.2p211")
        for name in view.system_tokens
    )
    assert any(name.startswith("edge:H-O:single") for name in view.system_tokens)
    assert _tanimoto(view.wl_tokens, view.wl_tokens) == pytest.approx(1.0)
