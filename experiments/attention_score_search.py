"""
attention_score_search.py - Small robustness pilot for attention score variants.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .robustness_benchmark import RobustnessBenchmarkConfig, run_robustness_benchmark
from .small_modelnet_experiment import DEFAULT_CLASS_PROTOCOL


DEFAULT_SCORE_SEARCH_METHODS: Tuple[str, ...] = (
    "came",
    "came_non_geometric_fusion_reg",
    "came_normalized_geometric_attention",
    "came_geom_coeff_mix",
)


@dataclass
class AttentionScoreSearchConfig:
    methods: Sequence[str] = DEFAULT_SCORE_SEARCH_METHODS
    data_root: Optional[str] = None
    class_protocol: str = DEFAULT_CLASS_PROTOCOL
    num_points: int = 128
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 4
    num_epochs: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    equiv_loss_weight: float = 1e-4
    aux_loss_weight: float = 0.1
    dropout: float = 0.0
    device: Optional[str] = None
    artifact_root: str = "artifacts/attention_score_search"
    val_samples_per_class: int = 5
    train_samples_per_class: int = 40


def _sort_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["mean_shift_accuracy"]),
            float(row["mean_accuracy_drop"]),
            float(row["mean_prediction_drift"]),
            -float(row["clean_accuracy"]),
        ),
    )


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "clean_accuracy",
        "mean_shift_accuracy",
        "mean_accuracy_drop",
        "mean_prediction_drift",
        "mean_prediction_agreement",
        "parameter_count",
        "train_runtime_seconds",
        "evaluation_runtime_seconds",
        "artifact_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def run_attention_score_search(config: AttentionScoreSearchConfig) -> Dict[str, object]:
    root = Path(config.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / f"search-{time.strftime('%Y%m%d-%H%M%S')}"
    artifact_dir.mkdir(parents=False, exist_ok=False)

    rows: List[Dict[str, object]] = []
    for method in config.methods:
        benchmark_config = RobustnessBenchmarkConfig(
            method=method,
            data_root=config.data_root,
            class_protocol=config.class_protocol,
            num_points=config.num_points,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            equiv_loss_weight=config.equiv_loss_weight,
            aux_loss_weight=config.aux_loss_weight,
            dropout=config.dropout,
            device=config.device,
            artifact_root=str(artifact_dir / "method_runs"),
            val_samples_per_class=config.val_samples_per_class,
            train_samples_per_class=config.train_samples_per_class,
        )
        result = run_robustness_benchmark(benchmark_config)
        metrics = result["metrics"]
        rows.append(
            {
                "method": method,
                "clean_accuracy": metrics["clean_accuracy"],
                "mean_shift_accuracy": metrics["mean_shift_accuracy"],
                "mean_accuracy_drop": metrics["mean_accuracy_drop"],
                "mean_prediction_drift": metrics["mean_prediction_drift"],
                "mean_prediction_agreement": metrics["mean_prediction_agreement"],
                "parameter_count": metrics["parameter_count"],
                "train_runtime_seconds": metrics["train_runtime_seconds"],
                "evaluation_runtime_seconds": metrics["evaluation_runtime_seconds"],
                "artifact_dir": result["artifact_dir"],
            }
        )

    ranked_rows = _sort_results(rows)
    (artifact_dir / "pilot_results.json").write_text(
        json.dumps({"config": asdict(config), "results": ranked_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(artifact_dir / "pilot_table.csv", ranked_rows)

    ranking_lines = ["# Attention Score Search Ranking", ""]
    for rank, row in enumerate(ranked_rows, start=1):
        ranking_lines.append(
            f"{rank}. `{row['method']}`: shift={row['mean_shift_accuracy']:.2f}, "
            f"drop={row['mean_accuracy_drop']:.2f}, drift={row['mean_prediction_drift']:.6f}, "
            f"clean={row['clean_accuracy']:.2f}"
        )
    (artifact_dir / "pilot_ranking.md").write_text("\n".join(ranking_lines) + "\n", encoding="utf-8")

    summary = [
        "# Attention Score Search Summary",
        "",
        f"- Methods: {', '.join(config.methods)}",
        f"- Epochs: {config.num_epochs}",
        f"- Num points: {config.num_points}",
        f"- Batch size: {config.batch_size}",
        "",
        "## Top Variant",
        "",
        f"- Method: {ranked_rows[0]['method']}" if ranked_rows else "- Method: none",
        f"- Mean shift accuracy: {ranked_rows[0]['mean_shift_accuracy']:.2f}" if ranked_rows else "- Mean shift accuracy: n/a",
        f"- Mean accuracy drop: {ranked_rows[0]['mean_accuracy_drop']:.2f}" if ranked_rows else "- Mean accuracy drop: n/a",
        f"- Mean prediction drift: {ranked_rows[0]['mean_prediction_drift']:.6f}" if ranked_rows else "- Mean prediction drift: n/a",
    ]
    (artifact_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    return {"artifact_dir": str(artifact_dir), "results": ranked_rows}


__all__ = ["AttentionScoreSearchConfig", "DEFAULT_SCORE_SEARCH_METHODS", "run_attention_score_search"]
