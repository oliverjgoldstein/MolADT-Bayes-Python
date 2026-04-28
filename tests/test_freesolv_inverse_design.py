from __future__ import annotations

from io import StringIO
import random
import runpy

import pytest

from experiments.freesolv_inverse_design import (
    DEFAULT_MODEL_DIR,
    FreeSolvBayesianPredictor,
    Prediction,
    _carbon_six_ring_seed,
    _remove_atom_if_terminal,
    _score_molecule,
    load_seed_molecules,
    molecular_formula,
    add_pi_ring_system,
    build_parser,
    main,
    run_inverse_design,
    write_result_molecule_files,
)
from moladt.chem.dietz import AtomId, mk_edge
from moladt.chem.mutable import MutableMolecule
from moladt.chem.validate import ValidationError, validate_molecule
from moladt.examples import diborane_pretty
from moladt.examples.sample_molecules import water


class FakePredictor:
    def predict(self, molecule) -> Prediction:
        heavy_atoms = sum(1 for atom in molecule.atoms.values() if atom.attributes.symbol.value != "H")
        return Prediction(mean=-float(heavy_atoms), sd=1.0)


class ExplodingPredictor:
    def predict(self, molecule) -> Prediction:
        raise AssertionError("invalid molecules must be rejected before scoring")


def test_cli_exposes_only_target_seed_molecule_plus_default_help() -> None:
    parser = build_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert options == {"-h", "--help", "--target", "--seed-molecule"}


def test_default_seed_molecule_is_water_and_seed_option_accepts_methane() -> None:
    parser = build_parser()

    default_args = parser.parse_args([])
    methane_args = parser.parse_args(["--seed-molecule", "methane"])

    assert default_args.seed_molecule == "water"
    assert methane_args.seed_molecule == "methane"
    assert len(load_seed_molecules(seed_molecule="water", n_seeds=5)) == 5


def test_tiny_smoke_run_prints_dietz_molecules() -> None:
    stream = StringIO()

    status = main(
        ["--target", "-5.0"],
        n_steps=8,
        n_seeds=1,
        top_k=2,
        predictor=FakePredictor(),
        stream=stream,
    )

    output = stream.getvalue()
    assert status == 0
    assert "Top generated molecules" in output
    assert "seed molecule: water" in output
    assert "bonding_systems:" in output


def test_generated_candidates_validate() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=12,
        n_seeds=2,
        top_k=5,
        predictor=FakePredictor(),
    )

    assert result.diagnostics.accepted_proposals >= 0
    for candidate in result.top_candidates:
        validate_molecule(candidate.molecule)


def test_fixed_water_seed_run_is_deterministic() -> None:
    first = run_inverse_design(
        target=-5.0,
        n_steps=16,
        n_seeds=3,
        top_k=5,
        predictor=FakePredictor(),
        seed_molecule="water",
    )
    second = run_inverse_design(
        target=-5.0,
        n_steps=16,
        n_seeds=3,
        top_k=5,
        predictor=FakePredictor(),
        seed_molecule="water",
    )

    first_summary = tuple(
        (candidate.score, molecular_formula(candidate.molecule), len(candidate.molecule.local_bonds))
        for candidate in first.top_candidates
    )
    second_summary = tuple(
        (candidate.score, molecular_formula(candidate.molecule), len(candidate.molecule.local_bonds))
        for candidate in second.top_candidates
    )

    assert first_summary == second_summary


def test_invalid_molecule_is_rejected_before_scoring() -> None:
    mutable = MutableMolecule.from_molecule(water)
    mutable.local_bonds.add(mk_edge(AtomId(1), AtomId(3)))

    with pytest.raises(ValidationError):
        _score_molecule(ExplodingPredictor(), mutable.freeze(), -5.0)


def test_add_pi_ring_system_never_duplicates_same_ring() -> None:
    molecule = _carbon_six_ring_seed()
    with_pi = add_pi_ring_system(molecule, random.Random(0))

    assert with_pi is not None
    validate_molecule(with_pi)
    assert add_pi_ring_system(with_pi, random.Random(1)) is None


def test_remove_terminal_atom_rejects_multiedge_dietz_participants() -> None:
    assert _remove_atom_if_terminal(diborane_pretty, AtomId(3)) is None


def test_result_writer_exports_importable_top_molecule_files(tmp_path) -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=4,
        n_seeds=1,
        top_k=2,
        predictor=FakePredictor(),
    )

    written = write_result_molecule_files(result, tmp_path)

    assert len(written.molecule_file_paths) == 2
    payload = runpy.run_path(str(written.molecule_file_paths[0]))
    validate_molecule(payload["molecule"])
    assert payload["rank"] == 1
    assert payload["seed_molecule"] == "water"
    assert payload["random_seed"] == 0
    assert payload["formula"]


def test_freesolv_predictor_loads_committed_freesolv_gp_parameters() -> None:
    predictor = FreeSolvBayesianPredictor.load()

    assert predictor.parameter_source_path == DEFAULT_MODEL_DIR / "details" / "model_coefficients.csv"
    assert predictor.alpha == -5.016285472314999
    assert predictor.signal_scale == 6.1231904871500005
    assert predictor.lengthscale == 3.81351448765
    assert predictor.sigma == 0.63860079588
