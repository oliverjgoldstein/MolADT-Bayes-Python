from __future__ import annotations

import importlib
import importlib.resources
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping

from ..chem.molecule import Molecule
from ..io.sdf import SDFRecord, read_sdf_records
from .descriptors import MolecularDescriptors, compute_descriptors, log1p_positive

POSTERIOR_PARAMETER_NAMES = (
    "intercept",
    "weight_coeff",
    "polar_coeff",
    "surface_coeff",
    "bond_coeff",
    "heavy_coeff",
    "halogen_coeff",
    "aromatic_ring_coeff",
    "aromatic_fraction_coeff",
    "rotatable_coeff",
    "weight_sq_coeff",
    "polar_sq_coeff",
    "surface_sq_coeff",
    "interaction_wp",
    "interaction_ws",
    "linear_scale",
    "quadratic_scale",
    "descriptor_scale",
)

SELECTED_PARAMETER_NAMES = (
    "intercept",
    "weight_coeff",
    "polar_coeff",
    "surface_coeff",
    "bond_coeff",
    "heavy_coeff",
    "halogen_coeff",
    "aromatic_ring_coeff",
    "aromatic_fraction_coeff",
    "rotatable_coeff",
    "linear_scale",
    "quadratic_scale",
    "descriptor_scale",
)

LOGP_PROPERTY_NAMES = ("logP", "PUBCHEM_XLOGP3")
NAME_PROPERTY_NAMES = ("molid", "PUBCHEM_IUPAC_NAME", "PUBCHEM_IUPAC_TRADITIONAL_NAME", "PUBCHEM_COMPOUND_CID")


@dataclass(frozen=True, slots=True)
class LogPObservation:
    name: str
    molecule: Molecule
    observed_logp: float


@dataclass(frozen=True, slots=True)
class NamedMolecule:
    name: str
    molecule: Molecule
    actual_logp: float | None = None


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    mean: float
    sd: float
    p05: float
    p50: float
    p95: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    name: str
    predicted_mean: float
    predicted_sd: float
    actual_logp: float | None
    residual: float | None


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    mae: float
    rmse: float
    residual_count: int
    largest_residuals: tuple[PredictionResult, ...]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    posterior_draw_count: int
    parameter_summaries: dict[str, PosteriorSummary]
    test_predictions: tuple[PredictionResult, ...]
    test_evaluation: EvaluationMetrics | None
    evaluation: EvaluationMetrics | None


PosteriorSamples = dict[str, tuple[float, ...]]


def read_logp_observations(path: str | Path, *, limit: int | None = None) -> list[LogPObservation]:
    observations: list[LogPObservation] = []
    for index, record in enumerate(read_sdf_records(path, limit=limit), start=1):
        logp = _record_logp(record)
        if logp is None:
            continue
        observations.append(LogPObservation(name=_record_name(record, index), molecule=record.molecule, observed_logp=logp))
    return observations


def read_named_molecules(path: str | Path) -> list[NamedMolecule]:
    named: list[NamedMolecule] = []
    for index, record in enumerate(read_sdf_records(path), start=1):
        named.append(NamedMolecule(name=_record_name(record, index), molecule=record.molecule, actual_logp=_record_logp(record)))
    return named


def sample_logp_model(
    observations: list[LogPObservation],
    *,
    num_chains: int = 4,
    num_samples: int = 1000,
    num_warmup: int | None = None,
    seed: int | None = None,
) -> PosteriorSamples:
    if not observations:
        raise ValueError("At least one observation is required for logP inference")
    try:
        stan = _import_stan_module()
    except ImportError as exc:
        raise RuntimeError(
            "PyStan 3 is required for infer-logp. Install the project dependencies with "
            "`python -m pip install -e \".[dev,legacy-pystan]\"` or install `pystan` directly."
        ) from exc
    _extend_httpstan_compile_timeout()
    data = _build_stan_data(observations)
    program = _stan_program()
    warmup = num_warmup if num_warmup is not None else max(50, num_samples)
    posterior = stan.build(program, data=data, random_seed=seed)
    fit = posterior.sample(num_chains=num_chains, num_samples=num_samples, num_warmup=warmup)
    draws = {name: tuple(_extract_draws(fit, name)) for name in POSTERIOR_PARAMETER_NAMES}
    draw_lengths = {len(values) for values in draws.values()}
    if draw_lengths != {len(next(iter(draws.values())))}:
        raise RuntimeError("Posterior draw lengths are inconsistent")
    return draws


def summarize_posterior(samples: PosteriorSamples) -> dict[str, PosteriorSummary]:
    return {name: summarize_draws(draws) for name, draws in samples.items()}


def summarize_draws(draws: tuple[float, ...]) -> PosteriorSummary:
    ordered = sorted(draws)
    return PosteriorSummary(
        mean=fmean(ordered),
        sd=0.0 if len(ordered) < 2 else pstdev(ordered),
        p05=_quantile(ordered, 0.05),
        p50=_quantile(ordered, 0.50),
        p95=_quantile(ordered, 0.95),
    )


def predict_logp(parameters: Mapping[str, float], descriptors: MolecularDescriptors) -> float:
    weight = descriptors.weight
    polar = descriptors.polar
    surface = descriptors.surface
    bond = descriptors.bond_order
    heavy_log = log1p_positive(descriptors.heavy_atoms)
    halogen_log = log1p_positive(descriptors.halogens)
    aromatic_ring_log = log1p_positive(descriptors.aromatic_rings)
    aromatic_fraction = descriptors.aromatic_atom_fraction
    rotatable_log = log1p_positive(descriptors.rotatable_bonds)
    return (
        parameters["intercept"]
        + parameters["weight_coeff"] * weight
        + parameters["polar_coeff"] * polar
        + parameters["surface_coeff"] * surface
        + parameters["bond_coeff"] * bond
        + parameters["heavy_coeff"] * heavy_log
        + parameters["halogen_coeff"] * halogen_log
        + parameters["aromatic_ring_coeff"] * aromatic_ring_log
        + parameters["aromatic_fraction_coeff"] * aromatic_fraction
        + parameters["rotatable_coeff"] * rotatable_log
        + parameters["weight_sq_coeff"] * weight * weight
        + parameters["polar_sq_coeff"] * polar * polar
        + parameters["surface_sq_coeff"] * surface * surface
        + parameters["interaction_wp"] * weight * polar
        + parameters["interaction_ws"] * weight * surface
    )


def predict_named_molecules(
    samples: PosteriorSamples,
    molecules: list[NamedMolecule],
    *,
    seed: int | None = None,
) -> tuple[PredictionResult, ...]:
    rng = random.Random(seed)
    draw_count = _posterior_draw_count(samples)
    predictions: list[PredictionResult] = []
    for molecule_index, named in enumerate(molecules):
        descriptors = compute_descriptors(named.molecule)
        predictive_draws: list[float] = []
        for draw_index in range(draw_count):
            params = {name: samples[name][draw_index] for name in POSTERIOR_PARAMETER_NAMES}
            mean_logp = predict_logp(params, descriptors)
            predictive_draws.append(rng.gauss(mean_logp, 0.2))
        predicted_mean = fmean(predictive_draws)
        predicted_sd = 0.0 if len(predictive_draws) < 2 else pstdev(predictive_draws)
        residual = None if named.actual_logp is None else predicted_mean - named.actual_logp
        predictions.append(
            PredictionResult(
                name=named.name if len(molecules) == 1 else f"{named.name}#{molecule_index + 1}",
                predicted_mean=predicted_mean,
                predicted_sd=predicted_sd,
                actual_logp=named.actual_logp,
                residual=residual,
            )
        )
    return tuple(predictions)


def evaluate_predictions(predictions: tuple[PredictionResult, ...]) -> EvaluationMetrics | None:
    residual_predictions = [prediction for prediction in predictions if prediction.residual is not None]
    if not residual_predictions:
        return None
    residuals = [prediction.residual for prediction in residual_predictions if prediction.residual is not None]
    mae = fmean(abs(residual) for residual in residuals)
    rmse = (fmean(residual * residual for residual in residuals)) ** 0.5
    ranked = tuple(sorted(residual_predictions, key=lambda prediction: abs(prediction.residual or 0.0), reverse=True)[:3])
    return EvaluationMetrics(mae=mae, rmse=rmse, residual_count=len(residuals), largest_residuals=ranked)


def run_logp_regression(
    train_observations: list[LogPObservation],
    test_molecules: list[NamedMolecule],
    *,
    evaluation_molecules: list[NamedMolecule] | None = None,
    num_chains: int = 4,
    num_samples: int = 1000,
    num_warmup: int | None = None,
    seed: int | None = None,
) -> InferenceResult:
    samples = sample_logp_model(
        train_observations,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        seed=seed,
    )
    summaries = summarize_posterior(samples)
    test_predictions = predict_named_molecules(samples, test_molecules, seed=seed)
    test_evaluation = evaluate_predictions(test_predictions)
    evaluation_predictions = (
        predict_named_molecules(samples, evaluation_molecules, seed=None if seed is None else seed + 1)
        if evaluation_molecules
        else ()
    )
    return InferenceResult(
        posterior_draw_count=_posterior_draw_count(samples),
        parameter_summaries=summaries,
        test_predictions=test_predictions,
        test_evaluation=test_evaluation,
        evaluation=evaluate_predictions(evaluation_predictions),
    )


def _record_logp(record: SDFRecord) -> float | None:
    for property_name in LOGP_PROPERTY_NAMES:
        value = record.property(property_name)
        if value is None:
            continue
        try:
            return float(value.splitlines()[0])
        except ValueError:
            continue
    return None


def _record_name(record: SDFRecord, index: int) -> str:
    for property_name in NAME_PROPERTY_NAMES:
        value = record.property(property_name)
        if value:
            return value.splitlines()[0]
    if record.title.strip():
        return record.title.strip()
    return f"record-{index}"


def _build_stan_data(observations: list[LogPObservation]) -> dict[str, Any]:
    descriptors = [compute_descriptors(observation.molecule) for observation in observations]
    return {
        "N": len(observations),
        "weight": [descriptor.weight for descriptor in descriptors],
        "polar": [descriptor.polar for descriptor in descriptors],
        "surface": [descriptor.surface for descriptor in descriptors],
        "bond_order": [descriptor.bond_order for descriptor in descriptors],
        "heavy_log": [log1p_positive(descriptor.heavy_atoms) for descriptor in descriptors],
        "halogen_log": [log1p_positive(descriptor.halogens) for descriptor in descriptors],
        "aromatic_ring_log": [log1p_positive(descriptor.aromatic_rings) for descriptor in descriptors],
        "aromatic_fraction": [descriptor.aromatic_atom_fraction for descriptor in descriptors],
        "rotatable_log": [log1p_positive(descriptor.rotatable_bonds) for descriptor in descriptors],
        "y": [observation.observed_logp for observation in observations],
    }


def _stan_program() -> str:
    return (Path(__file__).resolve().parent.parent / "stan" / "logp_regression.stan").read_text()


def _extract_draws(fit: Any, name: str) -> list[float]:
    try:
        values = fit[name]
        return list(_flatten_numeric(values))
    except Exception:
        if hasattr(fit, "to_frame"):
            try:
                frame = fit.to_frame()
            except Exception as exc:
                raise RuntimeError(f"Unable to extract posterior samples for {name}") from exc
            if name not in frame:
                raise RuntimeError(f"Posterior samples for {name} were not found")
            return [float(value) for value in frame[name].tolist()]
        raise


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        return _flatten_numeric(value.tolist())
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    raise TypeError(f"Unsupported posterior draw container: {type(value)!r}")


def _quantile(sorted_values: list[float] | tuple[float, ...], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot summarize empty draw sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _posterior_draw_count(samples: PosteriorSamples) -> int:
    return len(next(iter(samples.values())))


def _import_stan_module() -> Any:
    try:
        return importlib.import_module("stan")
    except PermissionError:
        import concurrent.futures

        class _NoopProcessPoolExecutor:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        os.environ["HTTPSTAN_DEBUG"] = "1"
        original_executor = concurrent.futures.ProcessPoolExecutor
        setattr(concurrent.futures, "ProcessPoolExecutor", _NoopProcessPoolExecutor)
        try:
            for module_name in list(sys.modules):
                if module_name == "stan" or module_name.startswith("stan.") or module_name == "httpstan" or module_name.startswith("httpstan."):
                    sys.modules.pop(module_name, None)
            return importlib.import_module("stan")
        finally:
            setattr(concurrent.futures, "ProcessPoolExecutor", original_executor)


def _extend_httpstan_compile_timeout(timeout_seconds: int = 30) -> None:
    try:
        httpstan_compile = importlib.import_module("httpstan.compile")
    except ImportError:
        return
    if getattr(httpstan_compile, "_moladt_timeout_patch", False):
        return

    def compile_with_longer_timeout(program_code: str, stan_model_name: str) -> tuple[str, str]:
        with importlib.resources.path("httpstan", "stanc") as stanc_binary:
            with tempfile.TemporaryDirectory(prefix="httpstan_") as tmpdir:
                filepath = Path(tmpdir) / f"{stan_model_name}.stan"
                filepath.write_text(program_code)
                run_args: Sequence[str | os.PathLike[str]] = [
                    stanc_binary,
                    "--name",
                    stan_model_name,
                    "--warn-pedantic",
                    "--print-cpp",
                    str(filepath),
                ]
                completed_process = subprocess.run(run_args, capture_output=True, timeout=timeout_seconds)
        stderr = completed_process.stderr.decode().strip()
        if completed_process.returncode != 0:
            raise ValueError(stderr)
        return completed_process.stdout.decode().strip(), stderr

    setattr(httpstan_compile, "compile", compile_with_longer_timeout)
    setattr(httpstan_compile, "_moladt_timeout_patch", True)
