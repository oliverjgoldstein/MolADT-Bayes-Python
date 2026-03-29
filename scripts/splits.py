from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import DEFAULT_SEED, DEFAULT_TRAIN_FRACTION, DEFAULT_VALID_FRACTION, PROCESSED_DATA_DIR, ensure_directory, write_json
from .features import FeatureTable


@dataclass(frozen=True, slots=True)
class ExportedDataset:
    dataset_name: str
    representation: str
    target_name: str
    feature_names: tuple[str, ...]
    feature_groups: dict[str, str]
    group_names: tuple[str, ...]
    group_ids: tuple[int, ...]
    rows: pd.DataFrame
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    mol_ids_train: tuple[str, ...]
    mol_ids_valid: tuple[str, ...]
    mol_ids_test: tuple[str, ...]
    metadata_path: Path
    feature_csv_path: Path


def deterministic_split_indices(
    row_count: int,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if row_count <= 2:
        raise ValueError("At least three rows are required for train/valid/test splitting")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    train_end = max(1, int(row_count * train_fraction))
    valid_end = min(row_count - 1, train_end + max(1, int(row_count * valid_fraction)))
    if valid_end >= row_count:
        valid_end = row_count - 1
    train_indices = np.sort(permutation[:train_end])
    valid_indices = np.sort(permutation[train_end:valid_end])
    test_indices = np.sort(permutation[valid_end:])
    if len(valid_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Split fractions produced an empty validation or test split")
    return train_indices, valid_indices, test_indices


def export_standardized_splits(
    feature_table: FeatureTable,
    *,
    dataset_name: str,
    representation: str,
    target_name: str,
    seed: int = DEFAULT_SEED,
) -> ExportedDataset:
    rows = feature_table.rows.copy()
    if rows.empty:
        raise ValueError(f"No rows available for {dataset_name}/{representation}")
    if "mol_id" in rows.columns:
        rows = rows.sort_values("mol_id").reset_index(drop=True)
    feature_names = feature_table.feature_names
    X = rows.loc[:, feature_names].to_numpy(dtype=float)
    y = rows[target_name].to_numpy(dtype=float)
    train_indices, valid_indices, test_indices = deterministic_split_indices(len(rows), seed=seed)
    X_train_raw = X[train_indices]
    X_valid_raw = X[valid_indices]
    X_test_raw = X[test_indices]
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std_safe = np.where(std == 0.0, 1.0, std)
    X_train = (X_train_raw - mean) / std_safe
    X_valid = (X_valid_raw - mean) / std_safe
    X_test = (X_test_raw - mean) / std_safe
    group_names = tuple(dict.fromkeys(feature_table.feature_groups[name] for name in feature_names))
    group_ids = tuple(group_names.index(feature_table.feature_groups[name]) + 1 for name in feature_names)

    prefix = f"{dataset_name}_{representation}"
    ensure_directory(PROCESSED_DATA_DIR)
    feature_csv_path = PROCESSED_DATA_DIR / f"{prefix}_features.csv"
    rows.to_csv(feature_csv_path, index=False)
    _write_split_frame(PROCESSED_DATA_DIR / f"{prefix}_X_train.csv", X_train, feature_names)
    _write_split_frame(PROCESSED_DATA_DIR / f"{prefix}_X_valid.csv", X_valid, feature_names)
    _write_split_frame(PROCESSED_DATA_DIR / f"{prefix}_X_test.csv", X_test, feature_names)
    _write_target_frame(PROCESSED_DATA_DIR / f"{prefix}_y_train.csv", y[train_indices], target_name)
    _write_target_frame(PROCESSED_DATA_DIR / f"{prefix}_y_valid.csv", y[valid_indices], target_name)
    _write_target_frame(PROCESSED_DATA_DIR / f"{prefix}_y_test.csv", y[test_indices], target_name)

    metadata = {
        "dataset": dataset_name,
        "representation": representation,
        "target_name": target_name,
        "seed": seed,
        "feature_names": list(feature_names),
        "feature_groups": feature_table.feature_groups,
        "group_names": list(group_names),
        "group_ids": list(group_ids),
        "train_mean": mean.tolist(),
        "train_std": std_safe.tolist(),
        "zero_variance_features": [name for name, value in zip(feature_names, std, strict=True) if value == 0.0],
        "split_indices": {
            "train": train_indices.tolist(),
            "valid": valid_indices.tolist(),
            "test": test_indices.tolist(),
        },
        "split_mol_ids": {
            "train": rows.loc[train_indices, "mol_id"].astype(str).tolist(),
            "valid": rows.loc[valid_indices, "mol_id"].astype(str).tolist(),
            "test": rows.loc[test_indices, "mol_id"].astype(str).tolist(),
        },
        "feature_csv": str(feature_csv_path.relative_to(PROCESSED_DATA_DIR.parent)),
    }
    metadata_path = write_json(PROCESSED_DATA_DIR / f"{prefix}_metadata.json", metadata)
    return ExportedDataset(
        dataset_name=dataset_name,
        representation=representation,
        target_name=target_name,
        feature_names=feature_names,
        feature_groups=feature_table.feature_groups,
        group_names=group_names,
        group_ids=group_ids,
        rows=rows,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y[train_indices],
        y_valid=y[valid_indices],
        y_test=y[test_indices],
        mol_ids_train=tuple(rows.loc[train_indices, "mol_id"].astype(str).tolist()),
        mol_ids_valid=tuple(rows.loc[valid_indices, "mol_id"].astype(str).tolist()),
        mol_ids_test=tuple(rows.loc[test_indices, "mol_id"].astype(str).tolist()),
        metadata_path=metadata_path,
        feature_csv_path=feature_csv_path,
    )


def _write_split_frame(path: Path, matrix: np.ndarray, feature_names: tuple[str, ...]) -> None:
    frame = pd.DataFrame(matrix, columns=list(feature_names))
    frame.to_csv(path, index=False)


def _write_target_frame(path: Path, target: np.ndarray, target_name: str) -> None:
    frame = pd.DataFrame({target_name: target})
    frame.to_csv(path, index=False)

