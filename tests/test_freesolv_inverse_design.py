from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import random
import runpy

import pytest

from experiments import freesolv_inverse_design as inverse_design
from experiments.freesolv_inverse_design import (
    FreeSolvBayesianPredictor,
    Prediction,
    _carbon_six_ring_seed,
    _ensure_plausible_freesolv_geometry,
    _find_gp_draws_path,
    _find_model_dir,
    _geometry_summary,
    _add_single_covalent_system,
    _molecule_key,
    _remove_atom_if_terminal,
    _score_molecule,
    _seed_atom,
    _seed_molecule,
    load_seed_molecules,
    molecular_formula,
    write_result_viewer_files,
    write_saved_inverse_design_viewer_file,
    add_pi_ring_system,
    build_parser,
    main,
    run_inverse_design,
    write_result_molecule_files,
)
from moladt.chem.dietz import AtomId, mk_edge
from moladt.chem.molecule import AtomicSymbol, molecule_edges
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


class FixedPredictor:
    def __init__(self, prediction: Prediction) -> None:
        self.prediction = prediction

    def predict(self, molecule) -> Prediction:
        return self.prediction


def test_cli_exposes_target_seed_and_optional_viewer_flags_plus_default_help() -> None:
    parser = build_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert options == {
        "-h",
        "--help",
        "--target",
        "--seed-molecule",
        "--open-viewer",
        "--viewer-count",
        "--view-results",
        "--viewer-output",
    }


def test_default_seed_molecule_is_freesolv_prior_and_seed_option_accepts_methane() -> None:
    parser = build_parser()

    default_args = parser.parse_args([])
    methane_args = parser.parse_args(["--seed-molecule", "methane"])

    assert default_args.seed_molecule == "freesolv-prior"
    assert default_args.viewer_count == 10
    assert methane_args.seed_molecule == "methane"
    assert len(load_seed_molecules(seed_molecule="water", n_seeds=5)) == 5
    prior_seeds = load_seed_molecules(
        seed_molecule="freesolv-prior",
        n_seeds=3,
        predictor=FakePredictor(),
        target=-5.0,
    )
    assert len(prior_seeds) == 3
    assert all(validate_molecule(molecule) == molecule for molecule in prior_seeds)


def test_tiny_smoke_run_prints_dietz_molecules() -> None:
    stream = StringIO()

    status = main(
        ["--target", "-5.0", "--seed-molecule", "water"],
        n_steps=8,
        n_seeds=1,
        top_k=2,
        predictor=FakePredictor(),
        stream=stream,
    )

    output = stream.getvalue()
    assert status == 0
    assert "Top generated molecules by Bayesian credible score" in output
    assert "Bayesian credible score:" in output
    assert "seed molecule: water" in output
    assert "bonding_systems:" in output


def test_default_inverse_design_samples_from_freesolv_prior() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=0,
        n_seeds=2,
        top_k=2,
        predictor=FakePredictor(),
    )

    assert result.seed_molecule == "freesolv-prior"
    assert len(result.top_candidates) == 2
    assert all(validate_molecule(candidate.molecule) == candidate.molecule for candidate in result.top_candidates)


def test_generated_candidates_validate() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=12,
        n_seeds=2,
        top_k=5,
        predictor=FakePredictor(),
        seed_molecule="water",
    )

    assert result.diagnostics.accepted_proposals >= 0
    for candidate in result.top_candidates:
        validate_molecule(candidate.molecule)
        geometry = _geometry_summary(candidate.molecule)
        assert geometry["min_bond_length_angstrom"] is not None


def test_top_candidates_are_highest_bayesian_credible_score_generated_candidates() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=0,
        n_seeds=1,
        top_k=3,
        predictor=FakePredictor(),
        min_unique_valid_molecules=8,
        seed_molecule="water",
    )

    top_scores = [candidate.bayesian_credible_score_percent for candidate in result.top_candidates]
    generated_scores = [candidate.bayesian_credible_score_percent for candidate in result.generated_candidates]

    assert top_scores == sorted(generated_scores, reverse=True)[:3]


def test_generated_candidate_uniqueness_ignores_incidental_coordinates() -> None:
    first = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.H, 0.9, 0.0, 0.0),
            _seed_atom(3, AtomicSymbol.H, -0.2, 0.8, 0.0),
        ),
        ((1, 2), (1, 3)),
    )
    second = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.O, 10.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.H, 10.9, 0.0, 0.0),
            _seed_atom(3, AtomicSymbol.H, 9.8, 0.8, 0.0),
        ),
        ((1, 2), (1, 3)),
    )

    assert _molecule_key(first) == _molecule_key(second)


def test_generation_moves_construct_valid_candidates_without_rejection_loop() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=50,
        n_seeds=2,
        top_k=5,
        predictor=FakePredictor(),
        seed_molecule="water",
    )

    assert result.diagnostics.total_proposals == 100
    assert result.diagnostics.valid_proposals >= 90
    assert result.diagnostics.invalid_proposals <= 10


def test_inverse_design_can_require_minimum_unique_valid_molecules() -> None:
    stream = StringIO()
    result = run_inverse_design(
        target=-5.0,
        n_steps=0,
        n_seeds=1,
        top_k=3,
        predictor=FakePredictor(),
        min_unique_valid_molecules=3,
        progress_stream=stream,
        seed_molecule="water",
    )

    assert result.minimum_unique_valid_molecules == 3
    assert len(result.generated_candidates) == 3
    assert result.diagnostics.unique_valid_molecules_seen >= 4
    assert result.diagnostics.total_proposals > 0
    assert "Generated unique valid candidates: 1/3" in stream.getvalue()
    assert "Generated unique valid candidates: 3/3" in stream.getvalue()
    assert "elapsed " in stream.getvalue()
    assert "s/candidate" in stream.getvalue()


def test_inverse_design_minimum_unique_target_stops_before_planned_steps() -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=50,
        n_seeds=1,
        top_k=3,
        predictor=FakePredictor(),
        min_unique_valid_molecules=3,
        seed_molecule="water",
    )

    assert len(result.generated_candidates) == 3
    assert result.diagnostics.total_proposals < 50


def test_inverse_design_fails_if_minimum_unique_valid_molecules_exceeds_proposal_budget() -> None:
    with pytest.raises(RuntimeError):
        run_inverse_design(
            target=-5.0,
            n_steps=0,
            n_seeds=1,
            top_k=1,
            predictor=FakePredictor(),
            min_unique_valid_molecules=2,
            max_total_proposals=0,
            seed_molecule="water",
        )


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
        (candidate.score, molecular_formula(candidate.molecule), len(molecule_edges(candidate.molecule)))
        for candidate in first.top_candidates
    )
    second_summary = tuple(
        (candidate.score, molecular_formula(candidate.molecule), len(molecule_edges(candidate.molecule)))
        for candidate in second.top_candidates
    )

    assert first_summary == second_summary


def test_invalid_molecule_is_rejected_before_scoring() -> None:
    mutable = MutableMolecule.from_molecule(water)
    _add_single_covalent_system(mutable, mk_edge(AtomId(1), AtomId(3)))

    with pytest.raises(ValidationError):
        _score_molecule(ExplodingPredictor(), mutable.freeze(), -5.0)


def test_underfilled_freesolv_candidate_is_rejected_before_scoring() -> None:
    molecule = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.C, 0.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.H, 1.09, 0.0, 0.0),
        ),
        ((1, 2),),
    )

    with pytest.raises(ValidationError):
        _score_molecule(ExplodingPredictor(), molecule, -5.0)


def test_charged_freesolv_candidate_is_rejected_before_scoring() -> None:
    mutable = MutableMolecule.from_molecule(water)
    oxygen = mutable.atoms[AtomId(1)]
    mutable.atoms[AtomId(1)] = replace(oxygen, formal_charge=1)

    with pytest.raises(ValidationError):
        _score_molecule(ExplodingPredictor(), mutable.freeze(), -5.0)


def test_geometry_audit_detects_overlapping_freesolv_candidate_coordinates() -> None:
    molecule = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.F, 1.42, 0.0, 0.0),
            _seed_atom(3, AtomicSymbol.H, 1.42, 0.0, 0.0),
        ),
        ((1, 2), (1, 3)),
    )

    with pytest.raises(ValidationError, match="overlapping coordinates"):
        _ensure_plausible_freesolv_geometry(molecule)


def test_geometry_audit_detects_nonbonded_van_der_waals_overlap() -> None:
    molecule = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.O, 1.65, 0.0, 0.0),
            _seed_atom(3, AtomicSymbol.H, 0.167, 0.945, 0.0),
            _seed_atom(4, AtomicSymbol.H, 1.483, 0.945, 0.0),
        ),
        ((1, 2), (1, 3), (2, 4)),
    )

    with pytest.raises(ValidationError, match="Non-bonded atoms"):
        _ensure_plausible_freesolv_geometry(molecule)


def test_geometry_audit_detects_tight_freesolv_candidate_bond_angles() -> None:
    molecule = _seed_molecule(
        (
            _seed_atom(1, AtomicSymbol.O, 0.0, 0.0, 0.0),
            _seed_atom(2, AtomicSymbol.F, 1.90, 0.0, 0.0),
            _seed_atom(3, AtomicSymbol.Cl, 0.382, 2.167, 0.0),
        ),
        ((1, 2), (1, 3)),
    )

    with pytest.raises(ValidationError, match="implausibly tight"):
        _ensure_plausible_freesolv_geometry(molecule)


def test_score_prefers_more_credible_prediction_at_same_target_error() -> None:
    high_credibility = _score_molecule(FixedPredictor(Prediction(mean=-5.0, sd=1.0)), water, -5.0)
    uncertain = _score_molecule(FixedPredictor(Prediction(mean=-5.0, sd=6.0)), water, -5.0)

    assert high_credibility.score > uncertain.score
    assert high_credibility.bayesian_credible_score_percent > uncertain.bayesian_credible_score_percent
    assert high_credibility.predictive_sd < uncertain.predictive_sd


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
        seed_molecule="water",
    )

    written = write_result_molecule_files(result, tmp_path)

    assert len(written.molecule_file_paths) == 2
    source = written.molecule_file_paths[0].read_text(encoding="utf-8")
    assert "molecule = Molecule(" in source
    assert "validate_molecule(" not in source
    assert "atoms = {" not in source
    assert "mk_edge(" not in source
    assert "Edge(AtomId(" in source
    payload = runpy.run_path(str(written.molecule_file_paths[0]))
    validate_molecule(payload["molecule"])
    assert payload["rank"] == 1
    assert payload["seed_molecule"] == "water"
    assert payload["random_seed"] == 0
    assert payload["formula"]
    assert "bayesian_credible_score_percent" in payload
    assert (tmp_path / "generated_molecules.csv").exists()
    jsonl_records = (tmp_path / "generated_molecules.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl_records) == len(result.generated_candidates)
    generated_record = json.loads(jsonl_records[0])
    assert generated_record["rank"] == 1
    assert "bayesian_credible_score_percent" in generated_record
    assert generated_record["min_bond_length_angstrom"] > 0.0
    assert generated_record["max_bond_length_angstrom"] >= generated_record["min_bond_length_angstrom"]
    assert "min_bond_angle_degrees" in generated_record
    assert "atoms" in generated_record["molecule"]
    assert "systems" in generated_record["molecule"]


def test_result_writer_removes_stale_candidate_files(tmp_path) -> None:
    stale = tmp_path / "dietz_01_molecule.py"
    stale_viewer = tmp_path / "top_01_molecule.viewer.html"
    stale_collection_viewer = tmp_path / "top_molecules.viewer.html"
    stale.write_text("stale = True\n", encoding="utf-8")
    stale_viewer.write_text("stale", encoding="utf-8")
    stale_collection_viewer.write_text("stale", encoding="utf-8")
    result = run_inverse_design(
        target=-5.0,
        n_steps=0,
        n_seeds=1,
        top_k=1,
        predictor=FakePredictor(),
        seed_molecule="water",
    )

    write_result_molecule_files(result, tmp_path)

    if stale.exists():
        assert stale.read_text(encoding="utf-8") != "stale = True\n"
    assert not stale_viewer.exists()
    assert not stale_collection_viewer.exists()
    assert (tmp_path / "top_01_molecule.py").exists()


def test_result_writer_can_export_top_candidate_viewers(tmp_path: Path) -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=4,
        n_seeds=1,
        top_k=2,
        predictor=FakePredictor(),
        seed_molecule="water",
    )

    written = write_result_viewer_files(result, tmp_path, count=2)

    assert len(written.viewer_file_paths) == 1
    assert written.viewer_file_paths[0].name == "top_molecules.viewer.html"
    html = written.viewer_file_paths[0].read_text(encoding="utf-8")
    assert "moladt-viewer-collection-v1" in html
    assert "FreeSolv top #1" in html
    assert "FreeSolv top #2" in html


def test_saved_inverse_design_results_can_be_reopened_in_viewer(tmp_path: Path) -> None:
    result = run_inverse_design(
        target=-5.0,
        n_steps=4,
        n_seeds=1,
        top_k=2,
        predictor=FakePredictor(),
        seed_molecule="water",
    )
    write_result_molecule_files(result, tmp_path)

    viewer_path = write_saved_inverse_design_viewer_file(tmp_path, count=2)

    assert viewer_path == tmp_path / "top_molecules.viewer.html"
    html = viewer_path.read_text(encoding="utf-8")
    assert "moladt-viewer-collection-v1" in html
    assert "FreeSolv top #1" in html
    assert "credible score" in html


def test_freesolv_model_dir_uses_latest_run_directory(tmp_path, monkeypatch) -> None:
    old_run = _write_minimal_freesolv_model_run(tmp_path / "run_20240101_000000")
    latest_run = _write_minimal_freesolv_model_run(tmp_path / "run_20240102_000000")
    monkeypatch.setattr(inverse_design, "FREESOLV_RESULTS_DIR", tmp_path)

    assert _find_model_dir() == latest_run
    assert _find_model_dir() != old_run


def test_latest_freesolv_model_dir_fails_fast_when_artifacts_are_missing(tmp_path, monkeypatch) -> None:
    _write_minimal_freesolv_model_run(tmp_path / "run_20240101_000000")
    (tmp_path / "run_20240102_000000" / "details").mkdir(parents=True)
    monkeypatch.setattr(inverse_design, "FREESOLV_RESULTS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Latest FreeSolv run"):
        _find_model_dir()


def test_freesolv_predictor_loads_committed_freesolv_gp_parameters() -> None:
    predictor = FreeSolvBayesianPredictor.load()
    model_dir = _find_model_dir()

    assert predictor.parameter_source_path == model_dir / "details" / "model_coefficients.csv"
    assert predictor.draw_source_path == _find_gp_draws_path(model_dir)
    assert predictor.signal_scale > 0.0
    assert predictor.lengthscale > 0.0
    assert predictor.sigma > 0.0
    assert predictor.alpha_draws.shape == (2000,)
    assert predictor.draw_weights.shape == (2000, 513)


def _write_minimal_freesolv_model_run(run_dir):
    draws_dir = (
        run_dir
        / "details"
        / "stan_output"
        / "freesolv"
        / "moladt_featurized"
        / "bayes_gp_rbf_screened"
        / "laplace"
    )
    draws_dir.mkdir(parents=True)
    (run_dir / "details" / "model_coefficients.csv").write_text("parameter_name,posterior_mean\n", encoding="utf-8")
    (draws_dir / "draws.csv").write_text("alpha,signal_scale,lengthscale,sigma\n", encoding="utf-8")
    return run_dir
