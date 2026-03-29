from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .chem.molecule import pretty_print_molecule
from .chem.validate import validate_molecule
from .examples import get_manuscript_example
from .inference.logp import (
    NamedMolecule,
    SELECTED_PARAMETER_NAMES,
    read_logp_observations,
    read_named_molecules,
    run_logp_regression,
)
from .io.sdf import read_sdf_record
from .io.smiles import molecule_to_smiles, parse_smiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m moladt.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse and validate an SDF file")
    parse_parser.add_argument("path")

    parse_smiles_parser = subparsers.add_parser("parse-smiles", help="Parse and validate a SMILES string")
    parse_smiles_parser.add_argument("smiles")

    to_smiles_parser = subparsers.add_parser("to-smiles", help="Render an SDF molecule as a SMILES string")
    to_smiles_parser.add_argument("path")

    pretty_example_parser = subparsers.add_parser(
        "pretty-example",
        help="Render a manuscript-facing built-in example molecule",
    )
    pretty_example_parser.add_argument("name", choices=("diborane", "ferrocene"))

    infer_parser = subparsers.add_parser("infer-logp", help="Run Stan-backed Bayesian logP regression")
    infer_parser.add_argument("--train", required=True)
    infer_parser.add_argument("--test", required=True)
    infer_parser.add_argument("--db2", default="logp/DB2.sdf")
    infer_parser.add_argument("--train-limit", type=int, default=300)
    infer_parser.add_argument("--db2-limit", type=int, default=300)
    infer_parser.add_argument("--num-chains", type=int, default=4)
    infer_parser.add_argument("--num-samples", type=int, default=1000)
    infer_parser.add_argument("--num-warmup", type=int, default=None)
    infer_parser.add_argument("--seed", type=int, default=1)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parse":
        return _handle_parse(Path(args.path))
    if args.command == "parse-smiles":
        return _handle_parse_smiles(args.smiles)
    if args.command == "to-smiles":
        return _handle_to_smiles(Path(args.path))
    if args.command == "pretty-example":
        return _handle_pretty_example(args.name)
    if args.command == "infer-logp":
        return _handle_infer_logp(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


def _handle_parse(path: Path) -> int:
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    print(f"Title: {record.title or '(blank)'}")
    print(pretty_print_molecule(record.molecule))
    if record.properties:
        print("Properties:")
        for key in sorted(record.properties):
            print(f"  {key}: {record.properties[key]}")
    return 0


def _handle_parse_smiles(smiles_text: str) -> int:
    molecule = parse_smiles(smiles_text)
    validate_molecule(molecule)
    print(pretty_print_molecule(molecule))
    return 0


def _handle_to_smiles(path: Path) -> int:
    record = read_sdf_record(path)
    validate_molecule(record.molecule)
    print(molecule_to_smiles(record.molecule))
    return 0


def _handle_pretty_example(name: str) -> int:
    example = get_manuscript_example(name)
    validate_molecule(example.molecule)
    print(example.render())
    return 0


def _handle_infer_logp(args: argparse.Namespace) -> int:
    try:
        train = read_logp_observations(args.train, limit=args.train_limit)
        test_molecules = read_named_molecules(args.test)
        evaluation_molecules = read_logp_observations(args.db2, limit=args.db2_limit) if Path(args.db2).exists() else []
        evaluation_named = [
            NamedMolecule(
                name=observation.name,
                molecule=observation.molecule,
                actual_logp=observation.observed_logp,
            )
            for observation in evaluation_molecules
        ]
        result = run_logp_regression(
            train,
            test_molecules,
            evaluation_molecules=evaluation_named,
            num_chains=args.num_chains,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            seed=args.seed,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Posterior draws: {result.posterior_draw_count}")
    print("Posterior summaries:")
    for name in SELECTED_PARAMETER_NAMES:
        summary = result.parameter_summaries[name]
        print(
            f"  {name}: mean={summary.mean:.4f} sd={summary.sd:.4f} "
            f"p05={summary.p05:.4f} p50={summary.p50:.4f} p95={summary.p95:.4f}"
        )
    print("Test-set predictions:")
    for prediction in result.test_predictions:
        if prediction.actual_logp is not None and prediction.residual is not None:
            print(
                f"  {prediction.name}: predicted={prediction.predicted_mean:.4f}, "
                f"actual={prediction.actual_logp:.4f}, residual={prediction.residual:.4f}, "
                f"predictive_sd={prediction.predicted_sd:.4f}"
            )
        else:
            print(
                f"  {prediction.name}: predicted={prediction.predicted_mean:.4f}, "
                f"actual=(unknown), predictive_sd={prediction.predicted_sd:.4f}"
            )
    if result.test_evaluation is not None:
        print(
            f"Test-set metrics: MAE={result.test_evaluation.mae:.4f} "
            f"RMSE={result.test_evaluation.rmse:.4f} n={result.test_evaluation.residual_count}"
        )
    if result.evaluation is not None:
        print(
            f"DB2 evaluation: MAE={result.evaluation.mae:.4f} "
            f"RMSE={result.evaluation.rmse:.4f} n={result.evaluation.residual_count}"
        )
        if result.evaluation.largest_residuals:
            print("DB2 predicted vs actual (largest residuals):")
            for prediction in result.evaluation.largest_residuals:
                print(
                    f"  {prediction.name}: predicted={prediction.predicted_mean:.4f}, "
                    f"actual={prediction.actual_logp:.4f}, residual={prediction.residual:.4f}, "
                    f"predictive_sd={prediction.predicted_sd:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
