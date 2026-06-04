#!/usr/bin/env python
"""Find taxa-level drivers behind antibiotic longitudinal OT groups.

This script starts from the individual groups produced by
`antibiotic_longi_period_ot_grouping.py`, recomputes OT transport plans with
taxa labels attached to each embedding row, and aggregates transport cost to
taxa-level contribution tables.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from antibiotic_longi_period_ot_grouping import (
    PERIODS,
    build_embedding_index,
    compute_cost_matrix,
    find_sample_pkl,
    l2_normalize,
    load_metadata,
    select_period_sample_ids,
    stable_rng,
    to_numpy_embedding,
)

GROUP_STABLE = "low_BA__low_CA"
GROUP_RECOVERY_FAST = "high_BA__low_CA"
GROUP_PERSISTENT_SHIFT = "high_BA__high_CA"
GROUP_DELAYED_SHIFT = "low_BA__high_CA"

# Edit this block when debugging. The script can be run directly without
# passing many command-line arguments.
RUN_CONFIG = {
    "metadata_tsv": "/bgi-seq-model-2/datasets/zhangkexin/meta_index/metaphlan4/fine-tune/Antibiotic.intervention.longi/supp.tsv",
    "embedding_dir": "/bgi-seq-model-2/codes/zhangkexin/meta_index/output/llama/v4/pretrain/Antibiotic.intervention.longi/",
    "groups_csv": "/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/OT/C.last.cutoff70/individual_ot_groups.csv",
    "output_dir": "/bgi-seq-model-2/codes/zhangkexin/meta_index/experiment/results/network/1.8/Antibiotic.intervention.longi.complete/OT/C.last.cutoff70/ot_driver_taxa/",
    "individual_col": "subject",
    "sample_col": "library",
    "period_col": "canon",
    "time_col": "day",
    "embedding_key": "embedding",
    "taxa_key": "taxa",
    "metric": "cosine",
    "no_l2_normalize": False,
    "sinkhorn_reg": 0.05,
    "sinkhorn_max_iter": 1000,
    "sinkhorn_tol": 1e-7,
    "max_cost_entries": 25_000_000,
    "max_taxa_per_sample": None,
    "seed": 20260522,
    "top_pairs_per_individual": 100,
    "min_individuals_per_taxa": 3,
    "include_b_to_c": True,
    # If True, C_to_A and optional B_to_C use only the last C-period sample.
    "last_c_sample_for_ca": True,
}


@dataclass
class PeriodDistribution:
    sample_ids: List[str]
    embeddings: np.ndarray
    weights: np.ndarray
    taxa: List[str]
    n_taxa_original: int


def load_groups(groups_csv: Path) -> pd.DataFrame:
    """Read four-group labels from the previous OT grouping experiment."""

    if not groups_csv.exists():
        raise FileNotFoundError(f"groups csv does not exist: {groups_csv}")
    groups = pd.read_csv(groups_csv)
    required_cols = ["individual_id", "group", "ot_ba", "ot_ca"]
    missing_cols = [col for col in required_cols if col not in groups.columns]
    if missing_cols:
        raise ValueError(
            f"groups csv is missing columns: {missing_cols}. "
            "Expected output from antibiotic_longi_period_ot_grouping.py."
        )
    groups = groups[required_cols].copy()
    groups["individual_id"] = groups["individual_id"].astype(str).str.strip()
    return groups


def normalize_taxa(value: object) -> List[str]:
    """Normalize the pkl `taxa` field to a clean string list."""

    if isinstance(value, pd.Series):
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("taxa field should be a list/tuple/array aligned to embedding rows")
    taxa = [str(item).strip() for item in value]
    return [item if item else "unknown_taxa" for item in taxa]


def choose_embedding_payload(payload: dict, embedding_key: str, path: Path) -> object:
    """Select taxa embedding from pkl and tolerate the historical `eembedding` typo."""

    if embedding_key in payload:
        return payload[embedding_key]
    if embedding_key == "embedding" and "eembedding" in payload:
        return payload["eembedding"]
    raise KeyError(f"{path} does not contain embedding key {embedding_key!r}")


def load_sample_taxa_embedding(path: Path, embedding_key: str, taxa_key: str) -> Tuple[np.ndarray, List[str]]:
    """Load one sample's taxa names and row-aligned taxa embedding matrix."""

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} should contain a dict")
    if taxa_key not in payload:
        raise KeyError(f"{path} does not contain taxa key {taxa_key!r}")

    embedding = to_numpy_embedding(choose_embedding_payload(payload, embedding_key, path))
    taxa = normalize_taxa(payload[taxa_key])
    if len(taxa) != embedding.shape[0]:
        raise ValueError(
            f"{path} taxa length {len(taxa)} does not match embedding rows {embedding.shape[0]}"
        )
    return embedding, taxa


def maybe_downsample(
    matrix: np.ndarray,
    taxa: Sequence[str],
    max_rows: Optional[int],
    seed: int,
    key: str,
) -> Tuple[np.ndarray, List[str]]:
    """Deterministically downsample embedding rows while keeping taxa aligned."""

    if max_rows is None or matrix.shape[0] <= max_rows:
        return matrix, list(taxa)

    rng = stable_rng(seed, key)
    indices = rng.choice(matrix.shape[0], size=max_rows, replace=False)
    indices.sort()
    return matrix[indices], [taxa[index] for index in indices]


def build_period_distribution(
    sample_ids: Sequence[str],
    sample_to_pkl: Dict[str, Path],
    args: SimpleNamespace,
) -> PeriodDistribution:
    """Merge all samples from one individual-period into one weighted taxa cloud.

    The weighting rule matches the grouping experiment: each sample gets equal
    mass within the period, and taxa inside that sample split the sample mass.
    """

    embeddings_list: List[np.ndarray] = []
    weights_list: List[np.ndarray] = []
    taxa_list: List[str] = []
    kept_sample_ids: List[str] = []
    n_taxa_original = 0

    sample_mass = 1.0 / len(sample_ids)
    for sample_id in sample_ids:
        embedding, taxa = load_sample_taxa_embedding(
            sample_to_pkl[sample_id],
            args.embedding_key,
            args.taxa_key,
        )
        n_taxa_original += embedding.shape[0]

        embedding, taxa = maybe_downsample(
            embedding,
            taxa,
            args.max_taxa_per_sample,
            args.seed,
            sample_id,
        )
        if not args.no_l2_normalize:
            embedding = l2_normalize(embedding)

        taxa_weight = sample_mass / embedding.shape[0]
        embeddings_list.append(embedding)
        weights_list.append(np.full(embedding.shape[0], taxa_weight, dtype=np.float64))
        taxa_list.extend(taxa)
        kept_sample_ids.append(sample_id)

    embeddings = np.vstack(embeddings_list).astype(np.float32, copy=False)
    weights = np.concatenate(weights_list)
    weights = weights / weights.sum()
    return PeriodDistribution(
        sample_ids=kept_sample_ids,
        embeddings=embeddings,
        weights=weights,
        taxa=taxa_list,
        n_taxa_original=n_taxa_original,
    )


def sinkhorn_transport_plan(
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    cost: np.ndarray,
    reg: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float, int, float]:
    """Compute Sinkhorn OT and keep the full transport plan for attribution."""

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
    return transport_plan, ot_cost, iteration, err


def compute_transport(
    source: PeriodDistribution,
    target: PeriodDistribution,
    args: SimpleNamespace,
) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """Build cost matrix and transport plan for one period comparison."""

    cost = compute_cost_matrix(
        source.embeddings,
        target.embeddings,
        args.metric,
        args.max_cost_entries,
    )
    transport_plan, ot_cost, n_iter, err = sinkhorn_transport_plan(
        source.weights,
        target.weights,
        cost,
        args.sinkhorn_reg,
        args.sinkhorn_max_iter,
        args.sinkhorn_tol,
    )
    return transport_plan, cost, ot_cost, n_iter, err


def contribution_rows(
    individual_id: str,
    group: str,
    comparison: str,
    source: PeriodDistribution,
    target: PeriodDistribution,
    transport_plan: np.ndarray,
    cost: np.ndarray,
    ot_cost: float,
) -> List[dict]:
    """Collapse point-level transport cost to source/target taxa contribution.

    Source taxa contribution answers: which taxa in B/C carry the deviation from
    the target period? Target taxa contribution answers: which target-period
    baseline taxa receive costly transport mass?
    """

    contribution = transport_plan * cost
    source_contrib = contribution.sum(axis=1)
    target_contrib = contribution.sum(axis=0)
    source_mass = transport_plan.sum(axis=1)
    target_mass = transport_plan.sum(axis=0)

    rows: Dict[Tuple[str, str], dict] = {}

    for taxa, value, mass in zip(source.taxa, source_contrib, source_mass):
        key = ("source", taxa)
        row = rows.setdefault(
            key,
            {
                "individual_id": individual_id,
                "group": group,
                "comparison": comparison,
                "taxa_role": "source",
                "taxa": taxa,
                "contribution": 0.0,
                "mass": 0.0,
                "n_rows": 0,
                "ot_cost": ot_cost,
            },
        )
        row["contribution"] += float(value)
        row["mass"] += float(mass)
        row["n_rows"] += 1

    for taxa, value, mass in zip(target.taxa, target_contrib, target_mass):
        key = ("target", taxa)
        row = rows.setdefault(
            key,
            {
                "individual_id": individual_id,
                "group": group,
                "comparison": comparison,
                "taxa_role": "target",
                "taxa": taxa,
                "contribution": 0.0,
                "mass": 0.0,
                "n_rows": 0,
                "ot_cost": ot_cost,
            },
        )
        row["contribution"] += float(value)
        row["mass"] += float(mass)
        row["n_rows"] += 1

    result = list(rows.values())
    for row in result:
        row["relative_contribution"] = (
            row["contribution"] / ot_cost if ot_cost > 0 else 0.0
        )
        row["contribution_per_mass"] = (
            row["contribution"] / row["mass"] if row["mass"] > 0 else 0.0
        )
    return result


def top_pair_rows(
    individual_id: str,
    group: str,
    comparison: str,
    source: PeriodDistribution,
    target: PeriodDistribution,
    transport_plan: np.ndarray,
    cost: np.ndarray,
    ot_cost: float,
    top_n: int,
) -> List[dict]:
    """Keep the largest source-taxa to target-taxa OT contribution pairs."""

    if top_n <= 0:
        return []

    contribution = transport_plan * cost
    flat = contribution.ravel()
    if flat.size == 0:
        return []

    top_n = min(top_n, flat.size)
    indices = np.argpartition(flat, -top_n)[-top_n:]
    indices = indices[np.argsort(flat[indices])[::-1]]

    rows = []
    n_target = contribution.shape[1]
    for flat_index in indices:
        source_index = int(flat_index // n_target)
        target_index = int(flat_index % n_target)
        value = float(contribution[source_index, target_index])
        if value <= 0:
            continue
        rows.append(
            {
                "individual_id": individual_id,
                "group": group,
                "comparison": comparison,
                "source_taxa": source.taxa[source_index],
                "target_taxa": target.taxa[target_index],
                "contribution": value,
                "relative_contribution": value / ot_cost if ot_cost > 0 else 0.0,
                "transport_mass": float(transport_plan[source_index, target_index]),
                "cost": float(cost[source_index, target_index]),
                "ot_cost": ot_cost,
            }
        )
    return rows


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def mann_whitney_pvalue(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a == 0 or n_b == 0:
        return math.nan, math.nan

    values = np.concatenate([group_a, group_b]).astype(np.float64)
    ranks = rank_values(values)
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = float(np.sum(tie_counts**3 - tie_counts))
    n = n_a + n_b
    mean_u = n_a * n_b / 2.0
    variance = n_a * n_b / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0
    if variance <= 0:
        p_value = 1.0
    else:
        z = (u - mean_u + 0.5) / math.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))

    rank_biserial = (2.0 * u_a) / (n_a * n_b) - 1.0
    return p_value, rank_biserial


def bh_fdr(p_values: Iterable[float]) -> List[float]:
    p = np.asarray(list(p_values), dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.where(~np.isnan(p))[0]
    if valid.size == 0:
        return q.tolist()

    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    adjusted = ranked * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[order] = adjusted
    return q.tolist()


def compare_groups(
    contribution_df: pd.DataFrame,
    comparison: str,
    taxa_role: str,
    group_a: str,
    group_b: str,
    label: str,
    min_individuals_per_taxa: int,
) -> pd.DataFrame:
    """Rank taxa by group difference for a single OT comparison."""

    subset = contribution_df[
        (contribution_df["comparison"] == comparison)
        & (contribution_df["taxa_role"] == taxa_role)
        & (contribution_df["group"].isin([group_a, group_b]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    individuals = (
        subset[["individual_id", "group"]]
        .drop_duplicates()
        .sort_values(["group", "individual_id"])
    )
    taxa_values = sorted(subset["taxa"].dropna().unique())
    rows = []
    for taxa in taxa_values:
        taxa_subset = subset[subset["taxa"] == taxa]
        matrix = individuals.copy()
        values = taxa_subset[["individual_id", "relative_contribution"]].groupby(
            "individual_id", as_index=False
        )["relative_contribution"].sum()
        matrix = matrix.merge(values, on="individual_id", how="left")
        matrix["relative_contribution"] = matrix["relative_contribution"].fillna(0.0)

        values_a = matrix.loc[
            matrix["group"] == group_a, "relative_contribution"
        ].to_numpy(dtype=np.float64)
        values_b = matrix.loc[
            matrix["group"] == group_b, "relative_contribution"
        ].to_numpy(dtype=np.float64)
        nonzero_a = int(np.sum(values_a > 0))
        nonzero_b = int(np.sum(values_b > 0))
        if max(nonzero_a, nonzero_b) < min_individuals_per_taxa:
            continue

        p_value, rank_biserial = mann_whitney_pvalue(values_a, values_b)
        rows.append(
            {
                "driver_label": label,
                "comparison": comparison,
                "taxa_role": taxa_role,
                "taxa": taxa,
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": len(values_a),
                "n_group_b": len(values_b),
                "nonzero_group_a": nonzero_a,
                "nonzero_group_b": nonzero_b,
                "mean_group_a": float(np.mean(values_a)) if len(values_a) else math.nan,
                "mean_group_b": float(np.mean(values_b)) if len(values_b) else math.nan,
                "median_group_a": float(np.median(values_a)) if len(values_a) else math.nan,
                "median_group_b": float(np.median(values_b)) if len(values_b) else math.nan,
                "mean_diff_a_minus_b": float(np.mean(values_a) - np.mean(values_b)),
                "median_diff_a_minus_b": float(np.median(values_a) - np.median(values_b)),
                "mannwhitney_p": p_value,
                "rank_biserial_a_vs_b": rank_biserial,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_q"] = bh_fdr(result["mannwhitney_p"])
    result = result.sort_values(
        ["mean_diff_a_minus_b", "fdr_q", "mannwhitney_p"],
        ascending=[False, True, True],
    )
    return result


def compare_delta_groups(
    contribution_df: pd.DataFrame,
    early_comparison: str,
    late_comparison: str,
    taxa_role: str,
    group_a: str,
    group_b: str,
    label: str,
    min_individuals_per_taxa: int,
) -> pd.DataFrame:
    """Rank taxa by how much early deviation resolves at the late period.

    For recovery-fast taxa, the useful signal is high B->A contribution followed
    by lower C->A contribution, so this compares B_to_A - C_to_A between groups.
    """

    subset = contribution_df[
        (contribution_df["comparison"].isin([early_comparison, late_comparison]))
        & (contribution_df["taxa_role"] == taxa_role)
        & (contribution_df["group"].isin([group_a, group_b]))
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    individuals = (
        subset[["individual_id", "group"]]
        .drop_duplicates()
        .sort_values(["group", "individual_id"])
    )
    taxa_values = sorted(subset["taxa"].dropna().unique())
    rows = []
    for taxa in taxa_values:
        taxa_subset = subset[subset["taxa"] == taxa]
        values = (
            taxa_subset.groupby(["individual_id", "comparison"], as_index=False)[
                "relative_contribution"
            ]
            .sum()
            .pivot(index="individual_id", columns="comparison", values="relative_contribution")
            .reset_index()
        )
        matrix = individuals.merge(values, on="individual_id", how="left")
        for column in [early_comparison, late_comparison]:
            if column not in matrix.columns:
                matrix[column] = 0.0
            matrix[column] = matrix[column].fillna(0.0)
        matrix["delta_early_minus_late"] = matrix[early_comparison] - matrix[late_comparison]

        values_a = matrix.loc[
            matrix["group"] == group_a, "delta_early_minus_late"
        ].to_numpy(dtype=np.float64)
        values_b = matrix.loc[
            matrix["group"] == group_b, "delta_early_minus_late"
        ].to_numpy(dtype=np.float64)
        nonzero_a = int(
            np.sum(
                (matrix["group"] == group_a)
                & ((matrix[early_comparison] > 0) | (matrix[late_comparison] > 0))
            )
        )
        nonzero_b = int(
            np.sum(
                (matrix["group"] == group_b)
                & ((matrix[early_comparison] > 0) | (matrix[late_comparison] > 0))
            )
        )
        if max(nonzero_a, nonzero_b) < min_individuals_per_taxa:
            continue

        p_value, rank_biserial = mann_whitney_pvalue(values_a, values_b)
        rows.append(
            {
                "driver_label": label,
                "comparison": f"{early_comparison}_minus_{late_comparison}",
                "taxa_role": taxa_role,
                "taxa": taxa,
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": len(values_a),
                "n_group_b": len(values_b),
                "nonzero_group_a": nonzero_a,
                "nonzero_group_b": nonzero_b,
                "mean_group_a": float(np.mean(values_a)) if len(values_a) else math.nan,
                "mean_group_b": float(np.mean(values_b)) if len(values_b) else math.nan,
                "median_group_a": float(np.median(values_a)) if len(values_a) else math.nan,
                "median_group_b": float(np.median(values_b)) if len(values_b) else math.nan,
                "mean_diff_a_minus_b": float(np.mean(values_a) - np.mean(values_b)),
                "median_diff_a_minus_b": float(np.median(values_a) - np.median(values_b)),
                "mannwhitney_p": p_value,
                "rank_biserial_a_vs_b": rank_biserial,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr_q"] = bh_fdr(result["mannwhitney_p"])
    result = result.sort_values(
        ["mean_diff_a_minus_b", "fdr_q", "mannwhitney_p"],
        ascending=[False, True, True],
    )
    return result


def build_driver_tables(
    contribution_df: pd.DataFrame,
    args: SimpleNamespace,
) -> Dict[str, pd.DataFrame]:
    """Create matched source/target driver tables with the same definitions."""

    driver_tables: Dict[str, pd.DataFrame] = {}
    for taxa_role in ("source", "target"):
        prefix = f"{taxa_role}_driver_taxa"
        driver_tables[f"{prefix}_recovery_fast"] = compare_delta_groups(
            contribution_df,
            "B_to_A",
            "C_to_A",
            taxa_role,
            GROUP_RECOVERY_FAST,
            GROUP_PERSISTENT_SHIFT,
            f"{taxa_role}_recovery_fast_B_shift_to_C_resolution",
            args.min_individuals_per_taxa,
        )
        driver_tables[f"{prefix}_persistent_shift"] = compare_groups(
            contribution_df,
            "C_to_A",
            taxa_role,
            GROUP_PERSISTENT_SHIFT,
            GROUP_RECOVERY_FAST,
            f"{taxa_role}_persistent_C_shift",
            args.min_individuals_per_taxa,
        )
        driver_tables[f"{prefix}_delayed_shift"] = compare_groups(
            contribution_df,
            "C_to_A",
            taxa_role,
            GROUP_DELAYED_SHIFT,
            GROUP_STABLE,
            f"{taxa_role}_delayed_C_shift",
            args.min_individuals_per_taxa,
        )
        driver_tables[f"{prefix}_stable_reference"] = compare_groups(
            contribution_df,
            "C_to_A",
            taxa_role,
            GROUP_STABLE,
            GROUP_PERSISTENT_SHIFT,
            f"{taxa_role}_stable_reference_low_C_shift",
            args.min_individuals_per_taxa,
        )
        if args.include_b_to_c:
            driver_tables[f"{prefix}_b_to_c_volatility"] = compare_groups(
                contribution_df,
                "B_to_C",
                taxa_role,
                GROUP_PERSISTENT_SHIFT,
                GROUP_RECOVERY_FAST,
                f"{taxa_role}_b_to_c_volatility",
                args.min_individuals_per_taxa,
            )
    return driver_tables


def main() -> int:
    args = SimpleNamespace(**RUN_CONFIG)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args)
    groups = load_groups(Path(args.groups_csv))
    metadata = metadata[metadata["individual_id"].isin(set(groups["individual_id"]))].copy()
    group_map = groups.set_index("individual_id")["group"].to_dict()

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

    contribution_rows_all: List[dict] = []
    pair_rows_all: List[dict] = []
    ot_rows: List[dict] = []
    skipped_rows: List[dict] = []

    comparisons = [("B_to_A", "B", "A"), ("C_to_A", "C", "A")]
    if args.include_b_to_c:
        comparisons.append(("B_to_C", "B", "C"))

    for individual_id, individual_df in metadata.groupby("individual_id", sort=True):
        group = group_map.get(individual_id)
        if group is None:
            continue

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
            if not matched_sample_ids:
                continue
            try:
                period_dists[period] = build_period_distribution(
                    matched_sample_ids,
                    sample_to_pkl,
                    args,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "individual_id": individual_id,
                        "period": period,
                        "reason": repr(exc),
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

        for comparison, source_period, target_period in comparisons:
            try:
                source = period_dists[source_period]
                target = period_dists[target_period]
                transport_plan, cost, ot_cost, n_iter, err = compute_transport(
                    source,
                    target,
                    args,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "individual_id": individual_id,
                        "period": comparison,
                        "reason": repr(exc),
                    }
                )
                continue

            ot_rows.append(
                {
                    "individual_id": individual_id,
                    "group": group,
                    "comparison": comparison,
                    "source_period": source_period,
                    "target_period": target_period,
                    "ot_cost": ot_cost,
                    "sinkhorn_iter": n_iter,
                    "sinkhorn_err": err,
                    "n_source_taxa": len(source.taxa),
                    "n_target_taxa": len(target.taxa),
                    "cost_entries": int(cost.size),
                }
            )
            contribution_rows_all.extend(
                contribution_rows(
                    individual_id,
                    group,
                    comparison,
                    source,
                    target,
                    transport_plan,
                    cost,
                    ot_cost,
                )
            )
            pair_rows_all.extend(
                top_pair_rows(
                    individual_id,
                    group,
                    comparison,
                    source,
                    target,
                    transport_plan,
                    cost,
                    ot_cost,
                    args.top_pairs_per_individual,
                )
            )

    contribution_df = pd.DataFrame(contribution_rows_all)
    pair_df = pd.DataFrame(pair_rows_all)
    ot_df = pd.DataFrame(ot_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    unmatched_df = pd.DataFrame({"sample_id": unmatched_samples})

    contribution_df.to_csv(output_dir / "taxa_ot_contribution_by_individual.csv", index=False)
    pair_df.to_csv(output_dir / "taxa_pair_ot_contribution.csv", index=False)
    ot_df.to_csv(output_dir / "individual_ot_driver_recomputed.csv", index=False)
    skipped_df.to_csv(output_dir / "skipped_driver_taxa_individuals.csv", index=False)
    unmatched_df.to_csv(output_dir / "unmatched_driver_taxa_samples.csv", index=False)

    if contribution_df.empty:
        raise RuntimeError("No contribution rows were produced. Check skipped output files.")

    driver_tables = build_driver_tables(contribution_df, args)

    for name, table in driver_tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    print(f"groups csv: {args.groups_csv}")
    print(f"metadata rows used: {len(metadata)}")
    print(f"unique samples in metadata: {len(unique_samples)}")
    print(f"matched sample pkl files: {len(sample_to_pkl)}")
    print(f"unmatched samples: {len(unmatched_samples)}")
    print(f"individual comparisons processed: {len(ot_df)}")
    print(f"taxa contribution rows: {len(contribution_df)}")
    print(f"top taxa pair rows: {len(pair_df)}")
    print(f"outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
