from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from .freesolv_small_feature_gp import (
    DEFAULT_SPLIT_COUNT,
    MODEL_FEATURES,
    run_freesolv_small_feature_gp_splits,
    write_freesolv_small_feature_outputs,
)


DEFAULT_SEED_START = 0


def run_ablation(
    *,
    seeds: Sequence[int],
    output_dir: Path,
    noise_floor: float = 0.01,
    verbose: bool = False,
) -> dict[str, Path]:
    result = run_freesolv_small_feature_gp_splits(
        model_keys=tuple(MODEL_FEATURES),
        seeds=seeds,
        noise_floor=float(noise_floor),
        verbose=verbose,
    )
    return write_freesolv_small_feature_outputs(result, output_dir)


def _default_output_dir() -> Path:
    return Path("results") / "freesolv_ablation" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarking.freesolv_representation_ablation",
        description="Run the FreeSolv small-feature ablation: atom bag, SMILES graph, and full MolADT.",
    )
    parser.add_argument("--split-count", type=int, default=DEFAULT_SPLIT_COUNT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.01,
        help="Lower bound on the GP observation-noise variance after target standardization.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split_count <= 0:
        raise ValueError("--split-count must be positive")
    if args.noise_floor <= 0.0:
        raise ValueError("--noise-floor must be positive")
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.split_count)))
    paths = run_ablation(
        seeds=seeds,
        output_dir=args.output_dir,
        noise_floor=float(args.noise_floor),
        verbose=bool(args.verbose),
    )
    summary = pd.read_csv(paths["summary"])
    test_summary = summary.loc[summary["split"] == "test"].copy()
    print("FreeSolv small-feature representation ablation")
    print(f"  seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} splits)")
    print(f"  output_dir: {args.output_dir}")
    for row in test_summary.sort_values("feature_count").itertuples(index=False):
        print(
            f"  {row.model}: "
            f"RMSE {float(row.rmse_mean):.6f} +/- {float(row.rmse_std):.6f} kcal/mol "
            f"features={int(row.feature_count)}"
        )
    print(f"  summary: {paths['summary']}")
    print(f"  features: {paths['feature_manifest']}")
    print(f"  svg: {paths['svg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
