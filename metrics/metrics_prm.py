"""Metrics runner scaffolding for PRM quantitative evaluation."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.utils_metrics import (
    create_timestamped_run_dir,
    ensure_project_root_on_path,
    flatten_config,
    resolve_device,
    seed_everything,
)


def compute_statistical_summary(values: torch.Tensor) -> Dict[str, float]:
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "count": 0,
        }

    values_cpu = values.detach().cpu().float()
    count = int(values_cpu.numel())
    mean = float(values_cpu.mean().item())
    std = float(values_cpu.std(unbiased=False).item()) if count > 1 else 0.0
    values_np = values_cpu.numpy()
    p25 = float(np.percentile(values_np, 25))
    median = float(np.percentile(values_np, 50))
    p75 = float(np.percentile(values_np, 75))

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "p25": p25,
        "p75": p75,
        "count": count,
    }


def format_stats(stats: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_median": stats["median"],
        f"{prefix}_p25": stats["p25"],
        f"{prefix}_p75": stats["p75"],
        f"{prefix}_count": stats["count"],
    }


def save_histogram_plot(values: torch.Tensor, bins: int, output_path: Path, title: str, xlabel: str) -> None:
    if values.numel() == 0:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Warning: matplotlib not available; cannot save histogram {output_path.name}: {exc}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values_np = values.detach().cpu().numpy()

    plt.figure(figsize=(4, 3))
    plt.hist(values_np, bins=bins, color="#1f77b4", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_distribution_metrics(win_counts: torch.Tensor) -> Dict[str, float]:
    total = win_counts.sum().item()
    if total <= 0:
        return {
            "coverage": 0.0,
            "sparsity": 1.0,
            "entropy_norm": float("nan"),
            "kl_uniform": float("nan"),
        }

    prsm_count = win_counts.numel()
    coverage = (win_counts > 0).float().mean().item()
    sparsity = 1.0 - coverage

    probs = win_counts.float() / total
    entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
    entropy_norm = entropy / np.log(prsm_count) if prsm_count > 1 else float("nan")
    kl_uniform = (probs * torch.log(probs * prsm_count + 1e-12)).sum().item()

    return {
        "coverage": coverage,
        "sparsity": sparsity,
        "entropy_norm": entropy_norm,
        "kl_uniform": kl_uniform,
    }


def safe_mean(values: List[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def safe_std(values: List[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) == 0:
        return float("nan")
    if len(valid) == 1:
        return 0.0
    mean_val = sum(valid) / len(valid)
    variance = sum((value - mean_val) ** 2 for value in valid) / (len(valid) - 1)
    return float(math.sqrt(max(variance, 0.0)))


def pairwise_cosine(latents: torch.Tensor) -> torch.Tensor:
    if latents.numel() == 0 or latents.shape[0] < 2:
        return torch.empty(0, dtype=torch.float32)
    normalized = F.normalize(latents.float(), dim=1, eps=1e-8)
    similarity_matrix = normalized @ normalized.T
    triu_indices = torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)
    return similarity_matrix[triu_indices[0], triu_indices[1]]


def cross_cosine(anchor: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
    if anchor.numel() == 0 or negatives.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    anchor_norm = F.normalize(anchor.float(), dim=1, eps=1e-8)
    negatives_norm = F.normalize(negatives.float(), dim=1, eps=1e-8)
    scores = anchor_norm @ negatives_norm.T
    return scores.reshape(-1)


def approximate_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return float("nan")
    pos = pos_scores.reshape(-1, 1)
    neg = neg_scores.reshape(1, -1)
    total_pairs = pos.numel() * neg.numel()
    if total_pairs == 0:
        return float("nan")
    greater = (pos > neg).sum().item()
    ties = (pos == neg).sum().item()
    auc = (greater + 0.5 * ties) / total_pairs
    return float(auc)


def compute_l2_selectivity_metrics(
    errors: torch.Tensor,
    latents: torch.Tensor,
    top_k: int,
    inter_negatives_k: int,
) -> Dict[str, float]:
    if errors.numel() == 0 or latents.numel() == 0:
        return {
            "l2_intra_cos_mean": float("nan"),
            "l2_inter_cos_mean": float("nan"),
            "l2_selectivity_delta": float("nan"),
            "l2_auc_opt": float("nan"),
        }

    errors_cpu = errors.detach().cpu().float()
    latents_cpu = latents.detach().cpu().float()

    if errors_cpu.dim() != 2 or latents_cpu.dim() != 3:
        return {
            "l2_intra_cos_mean": float("nan"),
            "l2_inter_cos_mean": float("nan"),
            "l2_selectivity_delta": float("nan"),
            "l2_auc_opt": float("nan"),
        }

    _, prsm_count = errors_cpu.shape
    top_latents_per_prsm: List[Optional[torch.Tensor]] = []

    for prsm_idx in range(prsm_count):
        prsm_errors = errors_cpu[:, prsm_idx]
        prsm_latents = latents_cpu[:, prsm_idx, :]

        finite_mask = torch.isfinite(prsm_errors)
        if finite_mask.sum() == 0:
            top_latents_per_prsm.append(None)
            continue

        prsm_errors = prsm_errors[finite_mask]
        prsm_latents = prsm_latents[finite_mask]

        effective_k = min(top_k, prsm_errors.shape[0])
        if effective_k <= 0:
            top_latents_per_prsm.append(None)
            continue

        top_values, top_indices = torch.topk(prsm_errors, k=effective_k, largest=False)
        selected_latents = prsm_latents[top_indices]
        top_latents_per_prsm.append(selected_latents)

    intra_means: List[float] = []
    inter_means: List[float] = []
    delta_values: List[float] = []
    auc_values: List[float] = []

    for prsm_idx in range(prsm_count):
        anchor_latents = top_latents_per_prsm[prsm_idx]
        if anchor_latents is None or anchor_latents.shape[0] == 0:
            continue

        intra_scores = pairwise_cosine(anchor_latents)
        intra_mean = float(intra_scores.mean().item()) if intra_scores.numel() > 0 else float("nan")

        negative_candidates: List[torch.Tensor] = []
        for other_idx, other_latents in enumerate(top_latents_per_prsm):
            if other_idx == prsm_idx or other_latents is None or other_latents.shape[0] == 0:
                continue
            negative_candidates.append(other_latents)

        if negative_candidates:
            negatives_concat = torch.cat(negative_candidates, dim=0)
            if negatives_concat.shape[0] > inter_negatives_k:
                start = (prsm_idx * inter_negatives_k) % negatives_concat.shape[0]
                indices = (torch.arange(inter_negatives_k) + start) % negatives_concat.shape[0]
                negatives_selected = negatives_concat[indices]
            else:
                negatives_selected = negatives_concat
            inter_scores = cross_cosine(anchor_latents, negatives_selected)
            inter_mean = float(inter_scores.mean().item()) if inter_scores.numel() > 0 else float("nan")
        else:
            inter_scores = torch.empty(0, dtype=torch.float32)
            inter_mean = float("nan")

        if not math.isnan(intra_mean) and not math.isnan(inter_mean):
            delta = intra_mean - inter_mean
        else:
            delta = float("nan")

        auc = approximate_auc(intra_scores, inter_scores)

        intra_means.append(intra_mean)
        inter_means.append(inter_mean)
        delta_values.append(delta)
        auc_values.append(auc)

    return {
        "l2_intra_cos_mean": safe_mean(intra_means),
        "l2_inter_cos_mean": safe_mean(inter_means),
        "l2_selectivity_delta": safe_mean(delta_values),
        "l2_auc_opt": safe_mean(auc_values),
    }


def compute_pearson_corr(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return float("nan")
    if vec_a.numel() != vec_b.numel():
        return float("nan")

    a = vec_a.detach().cpu().float().view(-1)
    b = vec_b.detach().cpu().float().view(-1)
    if a.numel() < 2:
        return float("nan")

    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.norm(a_centered) * torch.norm(b_centered)
    if denom.item() == 0:
        return float("nan")
    corr = torch.dot(a_centered, b_centered) / denom
    return float(torch.clamp(corr, -1.0, 1.0).item())


def compute_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return float("nan")
    if vec_a.numel() != vec_b.numel():
        return float("nan")

    a = vec_a.detach().cpu().float().view(-1)
    b = vec_b.detach().cpu().float().view(-1)
    denom = torch.norm(a) * torch.norm(b)
    if denom.item() == 0:
        return float("nan")
    cos = torch.dot(a, b) / denom
    return float(torch.clamp(cos, -1.0, 1.0).item())


def try_parse_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def run_win_rate_metrics(
    reconstruction_cache: Dict[int, Dict[str, torch.Tensor]],
    selected_indices: List[int],
    header: List[str],
    csv_path: Path,
) -> None:
    if not reconstruction_cache:
        print("No reconstruction cache available; skipping win-rate metrics.")
        return

    l2_columns = [col for col in header if col.startswith("l2_")]

    rows_to_write: List[Dict[str, Any]] = []
    progress = tqdm(selected_indices, desc="Step 2: Win-rate metrics", unit="img")
    for dataset_idx in progress:
        if dataset_idx not in reconstruction_cache:
            continue
        cache_entry = reconstruction_cache[dataset_idx]
        image_id = cache_entry.get("image_id")
        if image_id is None:
            continue

        l1_indices = cache_entry["l1_winner_indices"]
        l1_prsm_count = int(cache_entry["l1_prsm_count"].item())
        l1_win_counts = torch.bincount(l1_indices, minlength=l1_prsm_count)
        l1_metrics = compute_distribution_metrics(l1_win_counts)
        cache_entry["l1_win_counts"] = l1_win_counts

        row = {
            "image_id": image_id,
            "l1_cov": l1_metrics["coverage"],
            "l1_sparsity": l1_metrics["sparsity"],
            "l1_entropy_norm": l1_metrics["entropy_norm"],
            "l1_kl_uniform": l1_metrics["kl_uniform"],
            # Duplicate entropy for Step 3 documentation purposes
            "l1_heatmap_entropy_norm": l1_metrics["entropy_norm"],
        }

        if "l2_winner_indices" in cache_entry:
            l2_indices = cache_entry["l2_winner_indices"]
            l2_prsm_count = int(cache_entry["l2_prsm_count"].item())
            l2_win_counts = torch.bincount(l2_indices, minlength=l2_prsm_count)
            l2_metrics = compute_distribution_metrics(l2_win_counts)
            cache_entry["l2_win_counts"] = l2_win_counts
            row.update(
                {
                    "l2_cov": l2_metrics["coverage"],
                    "l2_sparsity": l2_metrics["sparsity"],
                    "l2_entropy_norm": l2_metrics["entropy_norm"],
                    "l2_kl_uniform": l2_metrics["kl_uniform"],
                    "l2_heatmap_entropy_norm": l2_metrics["entropy_norm"],
                }
            )
        else:
            for col in l2_columns:
                row[col] = float("nan")

        rows_to_write.append(row)

    if not rows_to_write:
        print("Win-rate metrics found no valid entries.")
        return

    existing_rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows.extend(reader)

    merged_rows: List[Dict[str, Any]] = []
    new_rows_map = {row["image_id"]: row for row in rows_to_write}
    for existing in existing_rows:
        image_id = existing.get("image_id")
        if image_id in new_rows_map:
            merged = existing.copy()
            merged.update(new_rows_map[image_id])
            merged_rows.append(merged)
        else:
            merged_rows.append(existing)

    header_set = list(existing_rows[0].keys()) if existing_rows else header
    for new_row in rows_to_write:
        if all(row.get("image_id") != new_row["image_id"] for row in merged_rows):
            merged_rows.append(new_row)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_set)
        writer.writeheader()
        writer.writerows(merged_rows)


def run_l2_selectivity_metrics(
    reconstruction_cache: Dict[int, Dict[str, torch.Tensor]],
    selected_indices: List[int],
    config: Dict,
    header: List[str],
    csv_path: Path,
) -> None:
    if not config["compute"].get("do_l2", False):
        return

    if not reconstruction_cache:
        print("No reconstruction cache available; skipping L2 selectivity metrics.")
        return

    top_k = config["l2_selectivity"]["top_k"]
    inter_negatives_k = config["l2_selectivity"]["inter_negatives_k"]
    selectivity_columns = {
        "l2_intra_cos_mean",
        "l2_inter_cos_mean",
        "l2_selectivity_delta",
        "l2_auc_opt",
    }

    rows_to_write: List[Dict[str, Any]] = []
    progress = tqdm(selected_indices, desc="Step 4: L2 selectivity", unit="img")
    for dataset_idx in progress:
        cache_entry = reconstruction_cache.get(dataset_idx)
        if not cache_entry or "l2_errors" not in cache_entry or "l2_latents" not in cache_entry:
            continue

        image_id = cache_entry.get("image_id")
        if image_id is None:
            continue

        metrics = compute_l2_selectivity_metrics(
            errors=cache_entry["l2_errors"],
            latents=cache_entry["l2_latents"],
            top_k=top_k,
            inter_negatives_k=inter_negatives_k,
        )

        latents_tensor = cache_entry["l2_latents"]
        if latents_tensor.numel() > 0:
            latent_vector = latents_tensor.float().mean(dim=(0, 1)).detach().cpu()
        else:
            latent_vector = torch.empty(0, dtype=torch.float32)
        cache_entry["l2_latent_mean"] = latent_vector

        row = {"image_id": image_id}
        row.update(metrics)
        rows_to_write.append(row)

        cache_entry.pop("l2_errors", None)
        cache_entry.pop("l2_latents", None)

    if not rows_to_write:
        print("L2 selectivity metrics found no valid entries.")
        return

    existing_rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows.extend(reader)

    merged_rows: List[Dict[str, Any]] = []
    new_rows_map = {row["image_id"]: row for row in rows_to_write}
    for existing in existing_rows:
        image_id = existing.get("image_id")
        if image_id in new_rows_map:
            merged = existing.copy()
            merged.update(new_rows_map[image_id])
            merged_rows.append(merged)
        else:
            merged_rows.append(existing)

    header_set = list(existing_rows[0].keys()) if existing_rows else header
    for new_row in rows_to_write:
        if all(row.get("image_id") != new_row["image_id"] for row in merged_rows):
            merged_rows.append(new_row)

    # Ensure selectivity columns exist in header for downstream processing
    for column in selectivity_columns:
        if column not in header_set:
            header_set.append(column)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_set)
        writer.writeheader()
        writer.writerows(merged_rows)


def run_temporal_stability_metrics(
    reconstruction_cache: Dict[int, Dict[str, torch.Tensor]],
    selected_indices: List[int],
    config: Dict,
    header: List[str],
    csv_path: Path,
) -> None:
    if not reconstruction_cache:
        print("No reconstruction cache available; skipping temporal stability metrics.")
        return

    if not selected_indices:
        return

    do_l2 = config["compute"].get("do_l2", False)

    l1_prev_map: Optional[torch.Tensor] = None
    l2_prev_counts: Optional[torch.Tensor] = None
    l2_prev_lattice: Optional[torch.Tensor] = None
    l2_prev_latent: Optional[torch.Tensor] = None

    rows_to_write: List[Dict[str, Any]] = []
    progress = tqdm(selected_indices, desc="Step 5: Temporal stability", unit="img")
    for dataset_idx in progress:
        cache_entry = reconstruction_cache.get(dataset_idx)
        if not cache_entry:
            continue

        image_id = cache_entry.get("image_id")
        if image_id is None:
            continue

        row: Dict[str, Any] = {"image_id": image_id}

        l1_map_sum = cache_entry.get("l1_activation_map_sum")
        l1_corr = compute_pearson_corr(l1_prev_map, l1_map_sum)
        row["l1_temporal_corr_r"] = l1_corr
        if l1_map_sum is not None:
            l1_prev_map = l1_map_sum.reshape(-1).clone()

        if do_l2:
            current_counts = cache_entry.get("l2_win_counts")
            row["l2_temporal_corr_win"] = compute_pearson_corr(l2_prev_counts, current_counts)
            if current_counts is not None:
                l2_prev_counts = current_counts.float().clone()

            lattice_shape = cache_entry.get("l2_rf_lattice_shape")
            winner_errors = cache_entry.get("l2_winner_errors")
            if (
                lattice_shape
                and winner_errors is not None
                and isinstance(lattice_shape, (tuple, list))
                and len(lattice_shape) == 2
                and winner_errors.numel() == lattice_shape[0] * lattice_shape[1]
            ):
                lattice_map = winner_errors.reshape(lattice_shape[0], lattice_shape[1])
                row["l2_temporal_corr_lattice"] = compute_pearson_corr(
                    l2_prev_lattice,
                    lattice_map,
                )
                l2_prev_lattice = lattice_map.reshape(-1).clone()
            else:
                row["l2_temporal_corr_lattice"] = float("nan")

            latent_vector = cache_entry.get("l2_latent_mean")
            row["l2_temporal_cos_latent"] = compute_cosine_similarity(l2_prev_latent, latent_vector)
            if latent_vector is not None and latent_vector.numel() > 0:
                l2_prev_latent = latent_vector.clone()

        rows_to_write.append(row)

        cache_entry.pop("l1_activation_map_sum", None)
        cache_entry.pop("l2_winner_errors", None)

    if not rows_to_write:
        print("Temporal stability metrics found no valid entries.")
        return

    existing_rows: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows.extend(reader)

    merged_rows: List[Dict[str, Any]] = []
    new_rows_map = {row["image_id"]: row for row in rows_to_write}
    for existing in existing_rows:
        image_id = existing.get("image_id")
        if image_id in new_rows_map:
            merged = existing.copy()
            merged.update(new_rows_map[image_id])
            merged_rows.append(merged)
        else:
            merged_rows.append(existing)

    header_set = list(existing_rows[0].keys()) if existing_rows else header
    new_columns = [
        "l1_temporal_corr_r",
        "l2_temporal_corr_win",
        "l2_temporal_corr_lattice",
        "l2_temporal_cos_latent",
    ]
    for column in new_columns:
        if column not in header_set:
            header_set.append(column)

    for new_row in rows_to_write:
        if all(row.get("image_id") != new_row["image_id"] for row in merged_rows):
            merged_rows.append(new_row)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header_set)
        writer.writeheader()
        writer.writerows(merged_rows)


def append_summary_rows(csv_path: Path, header: List[str]) -> Dict[str, Dict[str, float]]:
    if not csv_path.exists():
        return {}

    with csv_path.open("r", newline="") as handle:
        reader = list(csv.DictReader(handle))

    if not reader:
        return {}

    data_rows = [row for row in reader if not str(row.get("image_id", "")).startswith("summary_")]

    numeric_columns = [column for column in header if column != "image_id"]

    summary_mean: Dict[str, float] = {}
    summary_std: Dict[str, float] = {}
    summary_count: Dict[str, int] = {}

    for column in numeric_columns:
        values = [try_parse_float(row.get(column)) for row in data_rows]
        filtered = [value for value in values if not math.isnan(value)]
        summary_mean[column] = safe_mean(filtered)
        summary_std[column] = safe_std(filtered)
        summary_count[column] = len(filtered)

    mean_row = {"image_id": "summary_mean"}
    std_row = {"image_id": "summary_std"}
    count_row = {"image_id": "summary_count"}

    for column in numeric_columns:
        mean_row[column] = summary_mean.get(column, float("nan"))
        std_row[column] = summary_std.get(column, float("nan"))
        count_row[column] = summary_count.get(column, 0)

    rewritten_rows = data_rows + [mean_row, std_row, count_row]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rewritten_rows)

    return {"mean": summary_mean, "std": summary_std, "count": summary_count}


def print_summary_overview(summary: Dict[str, Dict[str, float]], do_l2: bool) -> None:
    if not summary:
        return

    mean_values = summary.get("mean", {})
    std_values = summary.get("std", {})

    key_columns = [
        "l1_mse_mean",
        "l1_cov",
        "l1_temporal_corr_r",
        "l2_mse_mean",
        "l2_cov",
        "l2_selectivity_delta",
        "l2_temporal_corr_win",
    ]

    print("Summary (mean ± std across processed images):")
    for column in key_columns:
        if column.startswith("l2_") and not do_l2:
            continue
        mean_val = mean_values.get(column)
        std_val = std_values.get(column)
        if mean_val is None or math.isnan(mean_val):
            continue
        std_text = "nan" if std_val is None or math.isnan(std_val) else f"{std_val:.4f}"
        print(f"  {column}: {mean_val:.4f} ± {std_text}")

def compute_l1_reconstruction_metrics(
    model: torch.nn.Module,
    tiles: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    layer_1 = model.layers["layer_1"] if "layer_1" in model.layers else None
    if layer_1 is None:
        raise RuntimeError("Layer 1 is not available in the loaded PRM model.")

    errors_chunks: List[torch.Tensor] = []
    winners_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in torch.split(tiles, batch_size):
            if batch.numel() == 0:
                continue
            batch_device = batch.to(device)
            recon, _ = layer_1(batch_device)
            target = batch_device.unsqueeze(1).expand(-1, recon.size(1), -1)
            batch_errors = F.mse_loss(
                recon,
                target,
                reduction="none",
            ).mean(dim=2)
            errors_chunks.append(batch_errors.detach().cpu())
            winners_chunks.append(batch_errors.argmin(dim=1).detach().cpu())

    prsm_count = layer_1.prsm_count if hasattr(layer_1, "prsm_count") else (errors_chunks[0].shape[1] if errors_chunks else 0)
    if errors_chunks:
        errors = torch.cat(errors_chunks, dim=0)
        winner_errors = errors.min(dim=1).values
        winner_indices = torch.cat(winners_chunks, dim=0)
    else:
        errors = torch.empty((0, prsm_count), dtype=torch.float32)
        winner_errors = torch.empty(0, dtype=torch.float32)
        winner_indices = torch.empty(0, dtype=torch.long)

    stats = compute_statistical_summary(winner_errors)
    formatted_stats = format_stats(stats, "l1_mse")

    return {
        "stats": formatted_stats,
        "errors": errors,
        "winner_errors": winner_errors,
        "winner_indices": winner_indices,
    }


def compute_l2_reconstruction_metrics(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    internal_batch_size: int,
    act_map_batch_size: int,
    rf_shape: List[int],
) -> Dict[str, Any]:
    layer_2 = model.layers["layer_2"] if "layer_2" in model.layers else None
    if layer_2 is None:
        raise RuntimeError("Layer 2 is not available in the loaded PRM model.")

    image_device = image_tensor.to(device)
    activation_map = model.generate_l1_activation_map(
        image_device,
        internal_batch_size=act_map_batch_size,
    )
    activation_map = activation_map.detach()
    activation_map_sum = activation_map.sum(dim=2).detach().cpu()

    map_h, map_w, _ = activation_map.shape
    rf_h, rf_w = rf_shape
    lattice_h = max(1, map_h - rf_h + 1)
    lattice_w = max(1, map_w - rf_w + 1)

    activation_map_4d = activation_map.permute(2, 0, 1).unsqueeze(0)
    receptive_fields = F.unfold(
        activation_map_4d,
        kernel_size=tuple(rf_shape),
    )
    l2_inputs = receptive_fields.squeeze(0).transpose(0, 1)

    errors_chunks: List[torch.Tensor] = []
    winners_chunks: List[torch.Tensor] = []
    latents_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in torch.split(l2_inputs, internal_batch_size):
            if batch.numel() == 0:
                continue
            batch_device = batch.to(device)
            recon, latent = layer_2(batch_device)
            target = batch_device.unsqueeze(1).expand(-1, recon.size(1), -1)
            batch_errors = F.mse_loss(
                recon,
                target,
                reduction="none",
            ).mean(dim=2)
            errors_chunks.append(batch_errors.detach().cpu())
            winners_chunks.append(batch_errors.argmin(dim=1).detach().cpu())
            latents_chunks.append(latent.detach().cpu())

    prsm_count = layer_2.prsm_count if hasattr(layer_2, "prsm_count") else (errors_chunks[0].shape[1] if errors_chunks else 0)
    if errors_chunks:
        errors = torch.cat(errors_chunks, dim=0)
        winner_errors = errors.min(dim=1).values
        winner_indices = torch.cat(winners_chunks, dim=0)
        latents = torch.cat(latents_chunks, dim=0)
    else:
        errors = torch.empty((0, prsm_count), dtype=torch.float32)
        winner_errors = torch.empty(0, dtype=torch.float32)
        winner_indices = torch.empty(0, dtype=torch.long)
        latents = torch.empty((0, prsm_count, 0), dtype=torch.float32)

    stats = compute_statistical_summary(winner_errors)
    formatted_stats = format_stats(stats, "l2_mse")

    return {
        "stats": formatted_stats,
        "errors": errors,
        "winner_errors": winner_errors,
        "winner_indices": winner_indices,
        "latents": latents,
        "activation_map_sum": activation_map_sum,
        "rf_lattice_shape": (lattice_h, lattice_w),
    }


def build_reconstruction_header(config: Dict) -> List[str]:
    header = [
        "image_id",
        "l1_mse_mean",
        "l1_mse_std",
        "l1_mse_median",
        "l1_mse_p25",
        "l1_mse_p75",
        "l1_mse_count",
        "l1_cov",
        "l1_sparsity",
        "l1_entropy_norm",
        "l1_kl_uniform",
        "l1_heatmap_entropy_norm",
        "l1_temporal_corr_r",
    ]
    if config["compute"]["do_l2"]:
        header.extend(
            [
                "l2_mse_mean",
                "l2_mse_std",
                "l2_mse_median",
                "l2_mse_p25",
                "l2_mse_p75",
                "l2_mse_count",
                "l2_cov",
                "l2_sparsity",
                "l2_entropy_norm",
                "l2_kl_uniform",
                "l2_heatmap_entropy_norm",
                "l2_intra_cos_mean",
                "l2_inter_cos_mean",
                "l2_selectivity_delta",
                "l2_auc_opt",
                "l2_temporal_corr_win",
                "l2_temporal_corr_lattice",
                "l2_temporal_cos_latent",
            ]
        )
    return header


def initialize_csv(csv_path: Path, header: List[str]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()


def run_reconstruction_stage(
    model: torch.nn.Module,
    dataset,
    selected_indices: List[int],
    config: Dict,
    csv_path: Path,
    run_dir: Path,
    header: List[str],
    device: torch.device,
) -> Tuple[Dict[str, int], Dict[int, Dict[str, torch.Tensor]]]:
    if not selected_indices:
        print("No images selected for reconstruction metrics; skipping Step 1.")
        return {"processed": 0, "skipped": 0}, {}

    internal_batch_size = config["batching"]["internal_batch_size"]
    act_map_batch_size = config["batching"]["gen_l1_act_map_batch_size"]
    rf_shape = config["model"]["l2_receptive_field"]
    l2_enabled = config["compute"]["do_l2"]
    l2_columns = [col for col in header if col.startswith("l2_")]
    plots_enabled = config["compute"].get("do_plots", False)
    hist_bins = config["plots"].get("hist_bins", 40)

    if plots_enabled:
        l1_plot_dir = run_dir / "histograms" / "l1"
        l1_plot_dir.mkdir(parents=True, exist_ok=True)
        l2_plot_dir = None
        if l2_enabled:
            l2_plot_dir = run_dir / "histograms" / "l2"
            l2_plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        l1_plot_dir = None
        l2_plot_dir = None

    processed = 0
    skipped = 0
    per_image_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    progress = tqdm(selected_indices, desc="Step 1: Reconstruction errors", unit="img")
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        for dataset_idx in progress:
            try:
                dataset_item = dataset[dataset_idx]
            except Exception as exc:
                print(f"Warning: failed to load dataset index {dataset_idx}: {exc}")
                skipped += 1
                continue

            if (not dataset_item) or dataset_item[0] is None or dataset_item[1] is None:
                print(f"Warning: dataset index {dataset_idx} returned empty data. Skipping.")
                skipped += 1
                continue

            image_tensor, tiles, image_path = dataset_item
            image_id = Path(image_path).name if image_path else f"index_{dataset_idx}"

            tiles = tiles.to(torch.float32).contiguous()
            l1_results = compute_l1_reconstruction_metrics(
                model,
                tiles,
                device,
                internal_batch_size,
            )

            row: Dict[str, Any] = {"image_id": image_id}
            row.update(l1_results["stats"])
            row.setdefault("l1_temporal_corr_r", float("nan"))
            per_image_cache[dataset_idx] = {
                "l1_winner_indices": l1_results["winner_indices"],
                "l1_prsm_count": torch.tensor(int(l1_results["errors"].shape[1]), dtype=torch.int32),
                "image_id": image_id,
                "l1_activation_map_sum": None,
            }

            if plots_enabled and l1_plot_dir is not None:
                image_stem = Path(image_id).stem
                l1_plot_path = l1_plot_dir / f"{image_stem}_l1_hist.png"
                save_histogram_plot(
                    l1_results["winner_errors"],
                    bins=hist_bins,
                    output_path=l1_plot_path,
                    title=f"L1 errors: {image_id}",
                    xlabel="Winner MSE",
                )

            if l2_enabled:
                try:
                    l2_results = compute_l2_reconstruction_metrics(
                        model,
                        image_tensor.to(torch.float32),
                        device,
                        internal_batch_size,
                        act_map_batch_size,
                        rf_shape,
                    )
                    row.update(l2_results["stats"])
                    per_image_cache[dataset_idx]["l2_winner_indices"] = l2_results["winner_indices"]
                    per_image_cache[dataset_idx]["l2_prsm_count"] = torch.tensor(int(l2_results["errors"].shape[1]), dtype=torch.int32)
                    per_image_cache[dataset_idx]["l2_errors"] = l2_results["errors"]
                    per_image_cache[dataset_idx]["l2_latents"] = l2_results["latents"]
                    per_image_cache[dataset_idx]["l1_activation_map_sum"] = l2_results["activation_map_sum"]
                    per_image_cache[dataset_idx]["l2_winner_errors"] = l2_results["winner_errors"]
                    per_image_cache[dataset_idx]["l2_rf_lattice_shape"] = l2_results["rf_lattice_shape"]
                    row.setdefault("l2_intra_cos_mean", float("nan"))
                    row.setdefault("l2_inter_cos_mean", float("nan"))
                    row.setdefault("l2_selectivity_delta", float("nan"))
                    row.setdefault("l2_auc_opt", float("nan"))
                    row.setdefault("l2_temporal_corr_win", float("nan"))
                    row.setdefault("l2_temporal_corr_lattice", float("nan"))
                    row.setdefault("l2_temporal_cos_latent", float("nan"))

                    if plots_enabled and l2_plot_dir is not None:
                        image_stem = Path(image_id).stem
                        l2_plot_path = l2_plot_dir / f"{image_stem}_l2_hist.png"
                        save_histogram_plot(
                            l2_results["winner_errors"],
                            bins=hist_bins,
                            output_path=l2_plot_path,
                            title=f"L2 errors: {image_id}",
                            xlabel="Winner MSE",
                        )
                except RuntimeError as exc:
                    print(f"Warning: L2 metrics unavailable for {image_id}: {exc}")
                    for col in l2_columns:
                        row[col] = float("nan")

            if per_image_cache[dataset_idx].get("l1_activation_map_sum") is None:
                try:
                    activation_map = model.generate_l1_activation_map(
                        image_tensor.to(device=device, dtype=torch.float32),
                        internal_batch_size=act_map_batch_size,
                    )
                    per_image_cache[dataset_idx]["l1_activation_map_sum"] = activation_map.sum(dim=2).detach().cpu()
                except Exception as exc:
                    print(f"Warning: failed to compute L1 activation map for {image_id}: {exc}")
                    per_image_cache[dataset_idx]["l1_activation_map_sum"] = torch.empty(0, dtype=torch.float32)

            writer.writerow(row)
            handle.flush()
            processed += 1

    return {"processed": processed, "skipped": skipped}, per_image_cache


def build_arg_parser(default_project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRM metrics runner (setup stage)")
    parser.add_argument("--project-root", default=str(default_project_root))
    parser.add_argument("--data-root", default=str(default_project_root / "data"))
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--checkpoint-dir", default=str(default_project_root / "models" / "PRM" / "saved_models"))
    parser.add_argument("--l1-checkpoint", default=None)
    parser.add_argument("--l2-checkpoint", default=None)
    parser.add_argument("--reports-dir", default=str(default_project_root / "reports" / "metrics_prm"))

    parser.add_argument("--device", default=None, help="Force device selection (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do-l1", dest="do_l1", action="store_true")
    parser.add_argument("--no-do-l1", dest="do_l1", action="store_false")
    parser.set_defaults(do_l1=True)

    parser.add_argument("--do-l2", dest="do_l2", action="store_true")
    parser.add_argument("--no-do-l2", dest="do_l2", action="store_false")
    parser.set_defaults(do_l2=True)

    parser.add_argument("--do-plots", dest="do_plots", action="store_true")
    parser.set_defaults(do_plots=False)

    parser.add_argument("--internal-batch-size", type=int, default=377)
    parser.add_argument("--gen-l1-act-map-batch-size", type=int, default=377)

    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--select-every-k", type=int, default=1)
    parser.add_argument("--frame-skip", type=int, default=2)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--inter-negatives-k", type=int, default=20)

    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--tile-stride", type=int, default=8)
    parser.add_argument("--l2-receptive-field", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--layers-to-load", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--hist-bins", type=int, default=40)

    return parser


def build_runner_config(args: argparse.Namespace) -> Dict:
    project_root = Path(args.project_root).resolve()
    data_root = Path(args.data_root).resolve()
    data_csv = Path(args.data_csv).resolve() if args.data_csv else (data_root / "data.csv")
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    l1_checkpoint = Path(args.l1_checkpoint).resolve() if args.l1_checkpoint else (checkpoint_dir / "prm_layer1_latest.pth")
    l2_checkpoint = Path(args.l2_checkpoint).resolve() if args.l2_checkpoint else (checkpoint_dir / "prm_layer2_latest.pth")
    reports_dir = Path(args.reports_dir).resolve()

    config: Dict = {
        "paths": {
            "project_root": project_root,
            "data_root": data_root,
            "data_csv": data_csv,
            "checkpoint_dir": checkpoint_dir,
            "l1_checkpoint": l1_checkpoint,
            "l2_checkpoint": l2_checkpoint,
            "reports_dir": reports_dir,
        },
        "compute": {
            "do_l1": args.do_l1,
            "do_l2": args.do_l2,
            "do_plots": args.do_plots,
        },
        "batching": {
            "internal_batch_size": args.internal_batch_size,
            "gen_l1_act_map_batch_size": args.gen_l1_act_map_batch_size,
        },
        "subsampling": {
            "max_images": args.max_images,
            "select_every_k": max(1, args.select_every_k),
            "frame_skip": max(0, args.frame_skip),
        },
        "l2_selectivity": {
            "top_k": max(1, args.top_k),
            "inter_negatives_k": max(1, args.inter_negatives_k),
        },
        "model": {
            "tile_params": {
                "size": max(1, args.tile_size),
                "stride": max(1, args.tile_stride),
            },
            "l2_receptive_field": [max(1, args.l2_receptive_field[0]), max(1, args.l2_receptive_field[1])],
            "layers_to_load": args.layers_to_load,
        },
        "seed": args.seed,
        "device_override": args.device,
        "plots": {
            "hist_bins": max(1, args.hist_bins),
        },
    }
    return config


def create_dataset(config: Dict):
    from utils.dataset.image_tile_dataset import ImageTileDataset

    transform = T.Compose([T.ToTensor()])
    dataset = ImageTileDataset(
        csv_file=str(config["paths"]["data_csv"]),
        data_root=str(config["paths"]["data_root"]),
        transform=transform,
        frame_skip=config["subsampling"]["frame_skip"],
        tile_params=config["model"]["tile_params"],
    )
    return dataset


def compute_selected_indices(dataset_len: int, config: Dict) -> List[int]:
    select_every_k = config["subsampling"]["select_every_k"]
    indices = list(range(0, dataset_len, select_every_k))
    max_images = config["subsampling"]["max_images"]
    if max_images is not None and max_images > 0:
        indices = indices[:max_images]
    return indices


def load_model(config: Dict, device: torch.device) -> torch.nn.Module:
    from models.PRM.prm import PRM, FiringSelectionMethod

    model = PRM(
        tile_params=config["model"]["tile_params"],
        l2_receptive_field=tuple(config["model"]["l2_receptive_field"]),
        layers_to_load=config["model"]["layers_to_load"],
        win_rate_momentum=0.995,
        firing_std_factor=1.0,
        conscience_factor=15.0,
        firing_selection_method=FiringSelectionMethod.WinnerTakeAll,
    )

    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    l1_checkpoint = config["paths"]["l1_checkpoint"]
    if config["compute"]["do_l1"]:
        if l1_checkpoint.exists():
            model.load_layer_weights(1, str(l1_checkpoint), device)
        else:
            print(f"Warning: L1 checkpoint not found at {l1_checkpoint}. Proceeding without weights.")

    l2_checkpoint = config["paths"]["l2_checkpoint"]
    if config["compute"]["do_l2"]:
        if l2_checkpoint.exists():
            model.load_layer_weights(2, str(l2_checkpoint), device)
        else:
            print(f"Warning: L2 checkpoint not found at {l2_checkpoint}. L2 metrics will be disabled at runtime.")
            config["compute"]["do_l2"] = False

    return model.to(device)
def main() -> None:
    default_project_root = Path(__file__).resolve().parents[1]
    parser = build_arg_parser(default_project_root)
    args = parser.parse_args()

    config = build_runner_config(args)

    project_root = Path(config["paths"]["project_root"])
    ensure_project_root_on_path(str(project_root))

    print("Reminder: activate the conda environment 'torch' before running metrics.")

    seed_everything(config["seed"])
    device = resolve_device(config["device_override"])

    flat_config = flatten_config(config)
    print("Resolved configuration:")
    for key in sorted(flat_config):
        print(f"  {key}: {flat_config[key]}")
    print(f"Using device: {device}")

    dataset = create_dataset(config)
    selected_indices = compute_selected_indices(len(dataset), config)
    print(f"Dataset images available: {len(dataset)}")
    print(f"Images selected for metrics: {len(selected_indices)}")
    if selected_indices != sorted(selected_indices):
        print("Warning: selected indices are not ordered; temporal metrics may be affected.")
    else:
        print("Selected indices are in ascending order (required for temporal metrics).")

    reports_base = config["paths"]["reports_dir"]
    run_dir = Path(create_timestamped_run_dir(str(reports_base)))
    csv_path = run_dir / "metrics.csv"
    header = build_reconstruction_header(config)
    initialize_csv(csv_path, header)
    print(f"Created reports directory: {run_dir}")
    print(f"Initialized CSV at: {csv_path}")

    model = load_model(config, device)
    model.eval()
    print("Model loaded and set to eval mode.")

    stage_stats, reconstruction_cache = run_reconstruction_stage(
        model=model,
        dataset=dataset,
        selected_indices=selected_indices,
        config=config,
        csv_path=csv_path,
        run_dir=run_dir,
        header=header,
        device=device,
    )

    processed = stage_stats.get("processed", 0)
    skipped = stage_stats.get("skipped", 0)
    print(f"Reconstruction metrics processed for {processed} images (skipped {skipped}).")

    run_win_rate_metrics(
        reconstruction_cache=reconstruction_cache,
        selected_indices=selected_indices,
        header=header,
        csv_path=csv_path,
    )
    print("Win-rate uniformity metrics appended.")

    run_l2_selectivity_metrics(
        reconstruction_cache=reconstruction_cache,
        selected_indices=selected_indices,
        config=config,
        header=header,
        csv_path=csv_path,
    )
    print("L2 selectivity metrics appended.")

    run_temporal_stability_metrics(
        reconstruction_cache=reconstruction_cache,
        selected_indices=selected_indices,
        config=config,
        header=header,
        csv_path=csv_path,
    )
    print("Temporal stability metrics appended.")

    summary_stats = append_summary_rows(csv_path=csv_path, header=header)
    print_summary_overview(summary_stats, do_l2=config["compute"].get("do_l2", False))


if __name__ == "__main__":
    main()
