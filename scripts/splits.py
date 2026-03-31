from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import DEFAULT_SEED, DEFAULT_TRAIN_FRACTION, DEFAULT_VALID_FRACTION, PROCESSED_DATA_DIR, ensure_directory, write_json
from .features import FeatureTable, GeometricFeatureTable


@dataclass(frozen=True, slots=True)
class ExportedDataset:
    dataset_name: str
    representation: str
    target_name: str
    split_scheme: str
    source_row_count: int
    used_row_count: int
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


@dataclass(frozen=True, slots=True)
class GeometricDatasetSpec:
    dataset_name: str
    representation: str
    target_name: str
    split_scheme: str
    source_row_count: int
    used_row_count: int
    rows: pd.DataFrame
    atomic_numbers: tuple[np.ndarray, ...]
    coordinates: tuple[np.ndarray, ...]
    global_feature_names: tuple[str, ...]
    global_feature_groups: dict[str, str]
    global_features: np.ndarray | None
    train_indices: np.ndarray
    valid_indices: np.ndarray
    test_indices: np.ndarray
    metadata_path: Path
    feature_csv_path: Path | None

    @property
    def mol_ids_train(self) -> tuple[str, ...]:
        return tuple(self.rows.loc[self.train_indices, "mol_id"].astype(str).tolist())

    @property
    def mol_ids_valid(self) -> tuple[str, ...]:
        return tuple(self.rows.loc[self.valid_indices, "mol_id"].astype(str).tolist())

    @property
    def mol_ids_test(self) -> tuple[str, ...]:
        return tuple(self.rows.loc[self.test_indices, "mol_id"].astype(str).tolist())

    @property
    def y(self) -> np.ndarray:
        return self.rows.loc[:, self.target_name].to_numpy(dtype=float)


@dataclass(frozen=True, slots=True)
class SplitPartition:
    train_indices: np.ndarray
    valid_indices: np.ndarray
    test_indices: np.ndarray
    unused_indices: np.ndarray
    scheme: str
    source_row_count: int


def deterministic_split_indices(
    row_count: int,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partition = deterministic_split_partition(
        row_count,
        seed=seed,
        train_fraction=train_fraction,
        valid_fraction=valid_fraction,
    )
    return partition.train_indices, partition.valid_indices, partition.test_indices


def deterministic_split_partition(
    row_count: int,
    *,
    seed: int = DEFAULT_SEED,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    valid_fraction: float = DEFAULT_VALID_FRACTION,
    train_size: int | None = None,
    valid_size: int | None = None,
    test_size: int | None = None,
    scheme: str | None = None,
) -> SplitPartition:
    if row_count <= 2:
        raise ValueError("At least three rows are required for train/valid/test splitting")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    if (train_size, valid_size, test_size).count(None) not in {0, 3}:
        raise ValueError("Split sizes must be either all provided or all omitted")
    if train_size is None:
        train_end = max(1, int(row_count * train_fraction))
        valid_end = min(row_count - 1, train_end + max(1, int(row_count * valid_fraction)))
        if valid_end >= row_count:
            valid_end = row_count - 1
        train_indices = np.sort(permutation[:train_end])
        valid_indices = np.sort(permutation[train_end:valid_end])
        test_indices = np.sort(permutation[valid_end:])
        unused_indices = np.array([], dtype=int)
        split_scheme = scheme or f"fractional:{train_fraction:.3f}/{valid_fraction:.3f}/{1.0 - train_fraction - valid_fraction:.3f}"
    else:
        train_count = int(train_size)
        valid_count = int(valid_size)
        test_count = int(test_size)
        if min(train_count, valid_count, test_count) <= 0:
            raise ValueError("Exact split sizes must all be positive")
        total_used = train_count + valid_count + test_count
        if total_used > row_count:
            raise ValueError(f"Exact split sizes require {total_used} rows but only {row_count} are available")
        train_end = train_count
        valid_end = train_count + valid_count
        test_end = total_used
        train_indices = np.sort(permutation[:train_end])
        valid_indices = np.sort(permutation[train_end:valid_end])
        test_indices = np.sort(permutation[valid_end:test_end])
        unused_indices = np.sort(permutation[test_end:])
        split_scheme = scheme or f"exact:{train_count}/{valid_count}/{test_count}"
    if len(valid_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Split fractions produced an empty validation or test split")
    return SplitPartition(
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        unused_indices=unused_indices,
        scheme=split_scheme,
        source_row_count=row_count,
    )


def export_standardized_splits(
    feature_table: FeatureTable,
    *,
    dataset_name: str,
    representation: str,
    target_name: str,
    seed: int = DEFAULT_SEED,
    split_partition: SplitPartition | None = None,
) -> ExportedDataset:
    rows = feature_table.rows.copy()
    if rows.empty:
        raise ValueError(f"No rows available for {dataset_name}/{representation}")
    if "mol_id" in rows.columns:
        rows = rows.sort_values("mol_id").reset_index(drop=True)
    feature_names = feature_table.feature_names
    X = rows.loc[:, feature_names].to_numpy(dtype=float)
    y = rows[target_name].to_numpy(dtype=float)
    partition = split_partition or deterministic_split_partition(len(rows), seed=seed)
    train_indices = partition.train_indices
    valid_indices = partition.valid_indices
    test_indices = partition.test_indices
    unused_indices = partition.unused_indices
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
        "split_scheme": partition.scheme,
        "source_row_count": int(partition.source_row_count),
        "used_row_count": int(len(train_indices) + len(valid_indices) + len(test_indices)),
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
            "unused": unused_indices.tolist(),
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
        split_scheme=partition.scheme,
        source_row_count=int(partition.source_row_count),
        used_row_count=int(len(train_indices) + len(valid_indices) + len(test_indices)),
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


def export_geometric_splits(
    geometric_table: GeometricFeatureTable,
    *,
    dataset_name: str,
    representation: str,
    target_name: str,
    seed: int = DEFAULT_SEED,
    split_partition: SplitPartition | None = None,
) -> GeometricDatasetSpec:
    rows = geometric_table.rows.copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No rows available for {dataset_name}/{representation}")
    order = np.arange(len(rows), dtype=int)
    if "mol_id" in rows.columns:
        order = rows.sort_values("mol_id").index.to_numpy(dtype=int)
        rows = rows.iloc[order].reset_index(drop=True)
    partition = split_partition or deterministic_split_partition(len(rows), seed=seed)
    train_indices = partition.train_indices
    valid_indices = partition.valid_indices
    test_indices = partition.test_indices
    unused_indices = partition.unused_indices

    atomic_numbers = tuple(geometric_table.atomic_numbers[index] for index in order.tolist())
    coordinates = tuple(geometric_table.coordinates[index] for index in order.tolist())
    if len(atomic_numbers) != len(rows) or len(coordinates) != len(rows):
        raise ValueError("Geometric arrays must align with exported rows")
    global_features: np.ndarray | None = None
    feature_csv_path: Path | None = None
    if geometric_table.global_features is not None:
        raw_global = np.asarray(geometric_table.global_features, dtype=float)[order]
        if raw_global.shape[0] != len(rows):
            raise ValueError("Global feature matrix must align with exported rows")
        if raw_global.ndim != 2:
            raise ValueError("Global feature matrix must be two-dimensional")
        mean = raw_global[train_indices].mean(axis=0)
        std = raw_global[train_indices].std(axis=0)
        std_safe = np.where(std == 0.0, 1.0, std)
        global_features = (raw_global - mean) / std_safe
        feature_frame = rows.loc[:, ["mol_id", target_name]].copy()
        for feature_index, feature_name in enumerate(geometric_table.global_feature_names):
            feature_frame[feature_name] = global_features[:, feature_index]
        ensure_directory(PROCESSED_DATA_DIR)
        feature_csv_path = PROCESSED_DATA_DIR / f"{dataset_name}_{representation}_global_features.csv"
        feature_frame.to_csv(feature_csv_path, index=False)
        zero_variance_features = [
            name
            for name, value in zip(geometric_table.global_feature_names, std, strict=True)
            if value == 0.0
        ]
        train_mean = mean.tolist()
        train_std = std_safe.tolist()
    else:
        zero_variance_features = []
        train_mean = []
        train_std = []

    ensure_directory(PROCESSED_DATA_DIR)
    metadata = {
        "dataset": dataset_name,
        "representation": representation,
        "target_name": target_name,
        "seed": seed,
        "split_scheme": partition.scheme,
        "source_row_count": int(partition.source_row_count),
        "used_row_count": int(len(train_indices) + len(valid_indices) + len(test_indices)),
        "global_feature_names": list(geometric_table.global_feature_names),
        "global_feature_groups": geometric_table.global_feature_groups,
        "global_feature_train_mean": train_mean,
        "global_feature_train_std": train_std,
        "zero_variance_global_features": zero_variance_features,
        "split_indices": {
            "train": train_indices.tolist(),
            "valid": valid_indices.tolist(),
            "test": test_indices.tolist(),
            "unused": unused_indices.tolist(),
        },
        "split_mol_ids": {
            "train": rows.loc[train_indices, "mol_id"].astype(str).tolist(),
            "valid": rows.loc[valid_indices, "mol_id"].astype(str).tolist(),
            "test": rows.loc[test_indices, "mol_id"].astype(str).tolist(),
        },
        "feature_csv": str(feature_csv_path.relative_to(PROCESSED_DATA_DIR.parent)) if feature_csv_path is not None else None,
    }
    metadata_path = write_json(PROCESSED_DATA_DIR / f"{dataset_name}_{representation}_geometry_metadata.json", metadata)
    return GeometricDatasetSpec(
        dataset_name=dataset_name,
        representation=representation,
        target_name=target_name,
        split_scheme=partition.scheme,
        source_row_count=int(partition.source_row_count),
        used_row_count=int(len(train_indices) + len(valid_indices) + len(test_indices)),
        rows=rows,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        global_feature_names=geometric_table.global_feature_names,
        global_feature_groups=geometric_table.global_feature_groups,
        global_features=global_features,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        metadata_path=metadata_path,
        feature_csv_path=feature_csv_path,
    )


def _write_split_frame(path: Path, matrix: np.ndarray, feature_names: tuple[str, ...]) -> None:
    frame = pd.DataFrame(matrix, columns=list(feature_names))
    frame.to_csv(path, index=False)


def _write_target_frame(path: Path, target: np.ndarray, target_name: str) -> None:
    frame = pd.DataFrame({target_name: target})
    frame.to_csv(path, index=False)
