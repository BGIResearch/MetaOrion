#!/usr/bin/env python
"""Compute period-level OT distances for the antibiotic longitudinal cohort.

This script expects explicit column names from the command line. In this
project, the period column is `canon`, and its values look like A1, B5, C10.
Only the leading A/B/C letter is used as the period label.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PERIODS = ("A", "B", "C")


@dataclass
class PeriodDistribution:
    """One individual's merged embedding distribution in one period."""

    sample_ids: List[str]
    embeddings: np.ndarray
    weights: np.ndarray
    n_microbes_original: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute B->A and C->A OT distances from microbiome embeddings."
    )
    parser.add_argument("--metadata-tsv", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--individual-col", required=True)
    parser.add_argument("--sample-col", required=True)
    parser.add_argument("--period-col", required=True)
    parser.add_argument("--time-col", default="day")

    parser.add_argument("--embedding-key", default="embedding")
    parser.add_argument("--metric", choices=("cosine", "sqeuclidean"), default="cosine")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--sinkhorn-reg", type=float, default=0.05)
    parser.add_argument("--sinkhorn-max-iter", type=int, default=1000)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-7)
    parser.add_argument("--max-cost-entries", type=int, default=25_000_000)
    parser.add_argument(
        "--max-microbes-per-sample",
        type=int,
        default=None,
        help="Optional deterministic downsampling per sample before OT.",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument(
        "--last-c-sample-for-ca",
        action="store_true",
        help="Use only the last C-period sample when computing C->A; keep A/B unchanged.",
    )
    return parser.parse_args()


def canon_to_period(value: object) -> Optional[str]:
    """Convert canon values like A1, B5, C10 to A/B/C."""

    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    match = re.match(r"^([ABC])\d+$", text)
    if match is None:
        return None
    return match.group(1)


def canon_to_order(value: object) -> float:
    """Extract numeric order from canon labels such as C1, C3, C10."""

    if pd.isna(value):
        return math.nan
    text = str(value).strip().upper()
    match = re.match(r"^[ABC](\d+)$", text)
    if match is None:
        return math.nan
    return float(match.group(1))


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    """Read metadata and keep rows with valid individual/sample/canon values."""

    metadata_path = Path(args.metadata_tsv)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")

    df = pd.read_csv(metadata_path, sep="\t")

    required_cols = [args.individual_col, args.sample_col, args.period_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"missing columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    selected_cols = [args.individual_col, args.sample_col, args.period_col]
    time_col = getattr(args, "time_col", None)
    has_time_col = bool(time_col) and time_col in df.columns
    if has_time_col and time_col not in selected_cols:
        selected_cols.append(time_col)

    metadata = df[selected_cols].copy()
    rename_map = {
        args.individual_col: "individual_id",
        args.sample_col: "sample_id",
        args.period_col: "canon",
    }
    if has_time_col:
        rename_map[time_col] = "time_value"
    metadata = metadata.rename(columns=rename_map)
    if "time_value" not in metadata.columns:
        metadata["time_value"] = np.nan

    metadata["individual_id"] = metadata["individual_id"].astype(str).str.strip()
    metadata["sample_id"] = metadata["sample_id"].astype(str).str.strip()
    metadata["period"] = metadata["canon"].map(canon_to_period)
    metadata["canon_order"] = metadata["canon"].map(canon_to_order)
    metadata["time_order"] = pd.to_numeric(metadata["time_value"], errors="coerce")
    metadata["row_order"] = np.arange(len(metadata), dtype=int)

    usable = metadata[
        metadata["individual_id"].ne("")
        & metadata["sample_id"].ne("")
        & metadata["period"].isin(PERIODS)
    ].copy()

    if usable.empty:
        raw_canon_values = (
            metadata["canon"].dropna().astype(str).drop_duplicates().head(50).tolist()
        )
        raise ValueError(
            "No usable metadata rows after parsing canon values. "
            "Expected canon values like A1, B5, C10. "
            f"First raw canon values: {raw_canon_values}"
        )

    return usable


def sort_period_metadata(period_df: pd.DataFrame) -> pd.DataFrame:
    """Sort one individual-period by day first, then canon order, then row order."""

    if period_df["time_order"].notna().any():
        sort_cols = ["time_order", "canon_order", "row_order", "sample_id"]
    elif period_df["canon_order"].notna().any():
        sort_cols = ["canon_order", "row_order", "sample_id"]
    else:
        sort_cols = ["row_order", "sample_id"]
    return period_df.sort_values(sort_cols, kind="mergesort")


def select_period_sample_ids(
    period_df: pd.DataFrame,
    period: str,
    use_last_c_sample_for_ca: bool,
) -> List[str]:
    """Select sample ids for one period; optionally keep only the last C sample."""

    sorted_df = sort_period_metadata(period_df)
    if period == "C" and use_last_c_sample_for_ca:
        return sorted_df.tail(1)["sample_id"].astype(str).tolist()
    return sorted_df["sample_id"].astype(str).drop_duplicates().tolist()


def build_embedding_index(embedding_dir: Path) -> Dict[str, Path]:
    """Index pkl files by file stem, which should match sample/library id."""

    if not embedding_dir.exists():
        raise FileNotFoundError(f"embedding dir does not exist: {embedding_dir}")

    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for path in sorted(embedding_dir.rglob("*.pkl")):
        key = path.stem
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path

    if duplicates:
        duplicate_msg = {key: [str(p) for p in paths] for key, paths in duplicates.items()}
        raise ValueError(f"duplicated pkl stems found: {duplicate_msg}")
    if not index:
        raise FileNotFoundError(f"no pkl files found under {embedding_dir}")
    return index


def find_sample_pkl(sample_id: str, embedding_dir: Path, pkl_index: Dict[str, Path]) -> Optional[Path]:
    """Find the pkl file for one sample/library id."""

    direct_path = embedding_dir / f"{sample_id}.pkl"
    if direct_path.exists():
        return direct_path
    return pkl_index.get(sample_id)


def to_numpy_embedding(value: object) -> np.ndarray:
    """Convert numpy/torch/list embedding values to a 2D float32 array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"embedding should be 2D, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"empty embedding, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def load_sample_embedding(path: Path, embedding_key: str) -> np.ndarray:
    """Load the `embedding` matrix from one sample pkl."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} should contain a dict with key {embedding_key!r}")
    if embedding_key not in payload:
        raise KeyError(f"{path} does not contain key {embedding_key!r}")
    return to_numpy_embedding(payload[embedding_key])


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return np.random.default_rng((seed + offset) % (2**32 - 1))


def maybe_downsample(
    matrix: np.ndarray,
    max_rows: Optional[int],
    seed: int,
    key: str,
) -> np.ndarray:
    if max_rows is None or matrix.shape[0] <= max_rows:
        return matrix

    rng = stable_rng(seed, key)
    indices = rng.choice(matrix.shape[0], size=max_rows, replace=False)
    indices.sort()
    return matrix[indices]


def build_period_distribution(
    sample_ids: Sequence[str],
    sample_to_pkl: Dict[str, Path],
    args: argparse.Namespace,
) -> PeriodDistribution:
    """Merge all samples in one individual-period with equal sample mass."""

    embeddings_list: List[np.ndarray] = []
    weights_list: List[np.ndarray] = []
    kept_sample_ids: List[str] = []
    n_microbes_original = 0

    sample_mass = 1.0 / len(sample_ids)
    for sample_id in sample_ids:
        embedding = load_sample_embedding(sample_to_pkl[sample_id], args.embedding_key)
        n_microbes_original += embedding.shape[0]

        embedding = maybe_downsample(
            embedding,
            args.max_microbes_per_sample,
            args.seed,
            sample_id,
        )
        if not args.no_l2_normalize:
            embedding = l2_normalize(embedding)

        microbe_weight = sample_mass / embedding.shape[0]
        embeddings_list.append(embedding)
        weights_list.append(np.full(embedding.shape[0], microbe_weight, dtype=np.float64))
        kept_sample_ids.append(sample_id)

    embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
    weights = np.concatenate(weights_list)
    weights = weights / weights.sum()

    return PeriodDistribution(
        sample_ids=kept_sample_ids,
        embeddings=embeddings,
        weights=weights,
        n_microbes_original=n_microbes_original,
    )


def compute_cost_matrix(
    source: np.ndarray,
    target: np.ndarray,
    metric: str,
    max_entries: int,
) -> np.ndarray:
    """Build the point-to-point cost matrix for two period distributions."""

    entries = source.shape[0] * target.shape[0]
    if entries > max_entries:
        raise MemoryError(
            f"cost matrix is too large: {source.shape[0]} x {target.shape[0]} "
            f"= {entries:,} entries. Use --max-microbes-per-sample."
        )

    if metric == "cosine":
        cost = 1.0 - np.matmul(source, target.T)
        return np.clip(cost, 0.0, 2.0).astype(np.float32, copy=False)

    source_norm = np.sum(source * source, axis=1, keepdims=True)
    target_norm = np.sum(target * target, axis=1, keepdims=True).T
    cost = source_norm + target_norm - 2.0 * np.matmul(source, target.T)
    return np.maximum(cost, 0.0).astype(np.float32, copy=False)


def sinkhorn_ot_cost(
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    cost: np.ndarray,
    reg: float,
    max_iter: int,
    tol: float,
) -> Tuple[float, int, float]:
    """Compute entropic OT cost with the Sinkhorn-Knopp iterations."""

    if reg <= 0:
        raise ValueError("--sinkhorn-reg must be positive")

    a = source_weights.astype(np.float64)
    b = target_weights.astype(np.float64)
    a = a / a.sum()
    b = b / b.sum()

    exponent = np.maximum(-cost.astype(np.float64) / reg, -745.0)
    kernel = np.maximum(np.exp(exponent), 1e-300)

    u = np.ones_like(a)
    v = np.ones_like(b)
    err = math.inf

    for iteration in range(1, max_iter + 1):
        u = a / np.maximum(kernel @ v, 1e-300)
        v = b / np.maximum(kernel.T @ u, 1e-300)

        if iteration == 1 or iteration % 20 == 0 or iteration == max_iter:
            transported_a = u * (kernel @ v)
            err = float(np.linalg.norm(transported_a - a, ord=1))
            if err < tol:
                break

    transport_plan = u[:, None] * kernel * v[None, :]
    ot_cost = float(np.sum(transport_plan * cost))
    return ot_cost, iteration, err


def compute_ot(
    source: PeriodDistribution,
    target: PeriodDistribution,
    args: argparse.Namespace,
) -> Tuple[float, int, float, int]:
    cost = compute_cost_matrix(
        source.embeddings,
        target.embeddings,
        args.metric,
        args.max_cost_entries,
    )
    ot_cost, n_iter, err = sinkhorn_ot_cost(
        source.weights,
        target.weights,
        cost,
        args.sinkhorn_reg,
        args.sinkhorn_max_iter,
        args.sinkhorn_tol,
    )
    return ot_cost, n_iter, err, int(cost.size)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args)
    embedding_dir = Path(args.embedding_dir)
    pkl_index = build_embedding_index(embedding_dir)

    unique_samples = sorted(metadata["sample_id"].unique())
    sample_to_pkl: Dict[str, Path] = {}
    unmatched_samples: List[str] = []
    for sample_id in unique_samples:
        pkl_path = find_sample_pkl(sample_id, embedding_dir, pkl_index)
        if pkl_path is None:
            unmatched_samples.append(sample_id)
        else:
            sample_to_pkl[sample_id] = pkl_path

    metadata["pkl_path"] = metadata["sample_id"].map(
        lambda sample_id: str(sample_to_pkl.get(sample_id, ""))
    )
    metadata.to_csv(output_dir / "metadata_with_pkl_match.csv", index=False)
    pd.DataFrame({"sample_id": unmatched_samples}).to_csv(
        output_dir / "unmatched_samples.csv", index=False
    )

    result_rows: List[dict] = []
    period_rows: List[dict] = []
    skipped_rows: List[dict] = []

    for individual_id, individual_df in metadata.groupby("individual_id", sort=True):
        period_dists: Dict[str, PeriodDistribution] = {}

        for period in PERIODS:
            period_df = individual_df[individual_df["period"] == period]
            if period_df.empty:
                continue

            sample_ids = select_period_sample_ids(
                period_df,
                period,
                args.last_c_sample_for_ca,
            )
            matched_sample_ids = [sid for sid in sample_ids if sid in sample_to_pkl]
            missing_sample_ids = [sid for sid in sample_ids if sid not in sample_to_pkl]
            if not matched_sample_ids:
                continue

            try:
                dist = build_period_distribution(matched_sample_ids, sample_to_pkl, args)
            except Exception as exc:
                skipped_rows.append(
                    {
                        "individual_id": individual_id,
                        "period": period,
                        "reason": repr(exc),
                    }
                )
                continue

            period_dists[period] = dist
            period_rows.append(
                {
                    "individual_id": individual_id,
                    "period": period,
                    "n_sample": len(dist.sample_ids),
                    "n_sample_unmatched": len(missing_sample_ids),
                    "n_microbe": dist.embeddings.shape[0],
                    "n_microbe_original": dist.n_microbes_original,
                    "weight_sum": float(dist.weights.sum()),
                    "sample_ids": ";".join(dist.sample_ids),
                    "missing_sample_ids": ";".join(missing_sample_ids),
                }
            )

        missing_periods = [period for period in PERIODS if period not in period_dists]
        if missing_periods:
            skipped_rows.append(
                {
                    "individual_id": individual_id,
                    "period": "",
                    "reason": "missing periods: " + ",".join(missing_periods),
                }
            )
            continue

        try:
            ot_ba, iter_ba, err_ba, entries_ba = compute_ot(
                period_dists["B"], period_dists["A"], args
            )
            ot_ca, iter_ca, err_ca, entries_ca = compute_ot(
                period_dists["C"], period_dists["A"], args
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "individual_id": individual_id,
                    "period": "OT",
                    "reason": repr(exc),
                }
            )
            continue

        result_rows.append(
            {
                "individual_id": individual_id,
                "n_sample_A": len(period_dists["A"].sample_ids),
                "n_sample_B": len(period_dists["B"].sample_ids),
                "n_sample_C": len(period_dists["C"].sample_ids),
                "n_microbe_A": period_dists["A"].embeddings.shape[0],
                "n_microbe_B": period_dists["B"].embeddings.shape[0],
                "n_microbe_C": period_dists["C"].embeddings.shape[0],
                "n_microbe_original_A": period_dists["A"].n_microbes_original,
                "n_microbe_original_B": period_dists["B"].n_microbes_original,
                "n_microbe_original_C": period_dists["C"].n_microbes_original,
                "ot_ba": ot_ba,
                "ot_ca": ot_ca,
                "sinkhorn_iter_ba": iter_ba,
                "sinkhorn_iter_ca": iter_ca,
                "sinkhorn_err_ba": err_ba,
                "sinkhorn_err_ca": err_ca,
                "cost_entries_ba": entries_ba,
                "cost_entries_ca": entries_ca,
                "sample_ids_A": ";".join(period_dists["A"].sample_ids),
                "sample_ids_B": ";".join(period_dists["B"].sample_ids),
                "sample_ids_C": ";".join(period_dists["C"].sample_ids),
            }
        )

    pd.DataFrame(period_rows).to_csv(output_dir / "period_distribution_summary.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(output_dir / "skipped_individuals_or_periods.csv", index=False)

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError(
            "No complete individuals were processed. Check unmatched_samples.csv "
            "and skipped_individuals_or_periods.csv."
        )

    ba_cutoff_quantile = 0.30
    ba_cutoff = float(results["ot_ba"].quantile(ba_cutoff_quantile))
    ca_cutoff = float(results["ot_ca"].median())
    results["ba_level"] = np.where(results["ot_ba"] >= ba_cutoff, "high_BA", "low_BA")
    results["ca_level"] = np.where(results["ot_ca"] >= ca_cutoff, "high_CA", "low_CA")
    results["group"] = results["ba_level"] + "__" + results["ca_level"]
    results["ba_cutoff_quantile"] = ba_cutoff_quantile
    results["ba_cutoff"] = ba_cutoff
    results["ca_cutoff_quantile"] = 0.50
    results["ca_cutoff"] = ca_cutoff
    results.to_csv(output_dir / "individual_ot_groups.csv", index=False)

    group_counts = results.groupby("group").size().reset_index(name="n_individual")
    group_counts.to_csv(output_dir / "group_counts.csv", index=False)

    print(f"metadata file: {args.metadata_tsv}")
    print(f"metadata rows used: {len(metadata)}")
    print(f"unique samples in metadata: {len(unique_samples)}")
    print(f"matched sample pkl files: {len(sample_to_pkl)}")
    print(f"unmatched samples: {len(unmatched_samples)}")
    print(f"complete individuals processed: {len(results)}")
    print(f"OT B->A cutoff q{ba_cutoff_quantile:.2f}: {ba_cutoff:.6g}")
    print(f"OT C->A cutoff q0.50: {ca_cutoff:.6g}")
    print(group_counts.to_string(index=False))
    print(f"outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
