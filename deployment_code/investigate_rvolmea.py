#!/usr/bin/env python3
"""
Statistical investigation of rvolmea datasets.

Reads CSV files from ./new_data/rvolmea/ that contain columns:
  x, y, z       -> measured position over time
  gx, gy, gz    -> reference position over time

Computes per-file metrics (per-axis and 3D) and generates figures.
Outputs a results table (printed and saved to CSV) and plots saved under
  ./new_data/rvolmea/reports/<timestamp>/
"""

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import types
import json
import datetime as dt
from typing import Dict, Tuple, Optional, List

import numpy as np
# Robust pandas import: handle environments where optional compiled deps
# (numexpr/bottleneck) are incompatible with NumPy.
try:
    import pandas as pd
except Exception:
    if "numexpr" not in sys.modules:
        sys.modules["numexpr"] = types.ModuleType("numexpr")
    if "bottleneck" not in sys.modules:
        sys.modules["bottleneck"] = types.ModuleType("bottleneck")
    import pandas as pd
import matplotlib

# Use non-interactive backend for headless environments before importing pyplot
matplotlib.use("Agg")


BASE_DATA_DIR = Path(__file__).parent / "new_data"


def _ensure_columns(
    df: pd.DataFrame,
    required=("x", "y", "z", "gx", "gy", "gz"),
) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )
    # Reorder and select only required columns
    return df.loc[:, required].astype(float)


def load_all_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    def sort_key(p: Path):
        stem = p.stem
        return (stem.isdigit(), int(stem) if stem.isdigit() else stem)

    files = sorted(data_dir.glob("*.csv"), key=sort_key)
    datasets: Dict[str, pd.DataFrame] = {}
    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = _ensure_columns(df)
            datasets[fp.stem] = df
        except Exception as e:
            print(f"[WARN] Failed to load {fp.name}: {e}")
    if not datasets:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return datasets


def compute_metrics_for_df(df: pd.DataFrame) -> Dict[str, float]:
    # Errors (measured - ref)
    ex = df["x"].to_numpy() - df["gx"].to_numpy()
    ey = df["y"].to_numpy() - df["gy"].to_numpy()
    ez = df["z"].to_numpy() - df["gz"].to_numpy()
    e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

    def _skew_kurtosis(e: np.ndarray) -> Tuple[float, float]:
        if e.size < 4:
            return (np.nan, np.nan)
        m = np.mean(e)
        s = np.std(e)
        if s == 0:
            return (np.nan, np.nan)
        z = (e - m) / s
        sk = float(np.mean(z ** 3))
        ku = float(np.mean(z ** 4) - 3.0)
        return (sk, ku)

    def axis_stats(e: np.ndarray, prefix: str) -> Dict[str, float]:
        sk, ku = _skew_kurtosis(e)
        return {
            f"{prefix}_rmse": (
                float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan
            ),
            f"{prefix}_mae": float(np.mean(np.abs(e))) if e.size else np.nan,
            f"{prefix}_bias": float(np.mean(e)) if e.size else np.nan,
            f"{prefix}_std": float(np.std(e)) if e.size else np.nan,
            f"{prefix}_max_abs": (
                float(np.max(np.abs(e))) if e.size else np.nan
            ),
            f"{prefix}_p95_abs": (
                float(np.percentile(np.abs(e), 95)) if e.size else np.nan
            ),
            f"{prefix}_skew": sk,
            f"{prefix}_kurtosis": ku,
        }

    metrics: Dict[str, float] = {
        "n_samples": int(len(df)),
        # 3D composite metrics
        "rmse_3d": float(np.sqrt(np.mean(e3 ** 2))) if e3.size else np.nan,
        "mae_3d": float(np.mean(e3)) if e3.size else np.nan,
        "median_3d": float(np.median(e3)) if e3.size else np.nan,
        "p95_3d": float(np.percentile(e3, 95)) if e3.size else np.nan,
        "max_3d": float(np.max(e3)) if e3.size else np.nan,
        "iqr_3d": (
            float(
                np.percentile(e3, 75)
                - np.percentile(e3, 25)
            )
            if e3.size
            else np.nan
        ),
    }

    metrics.update(axis_stats(ex, "x"))
    metrics.update(axis_stats(ey, "y"))
    metrics.update(axis_stats(ez, "z"))

    # Reference magnitude (optional context)
    gx, gy, gz = df[["gx", "gy", "gz"]].to_numpy().T
    gnorm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    metrics.update(
        {
            "ref_span": (
                float(np.max(gnorm) - np.min(gnorm)) if gnorm.size else np.nan
            ),
            "ref_median": float(np.median(gnorm)) if gnorm.size else np.nan,
        }
    )

    return metrics


def summarize_across_files(
    results_df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    # Identify best and worst by rmse_3d
    best_idx = results_df["rmse_3d"].idxmin()
    worst_idx = results_df["rmse_3d"].idxmax()
    median_idx = (
        results_df["rmse_3d"].sort_values().index[len(results_df) // 2]
    )
    return {
        "best": {
            "file": str(best_idx),
            "rmse_3d": f"{results_df.loc[best_idx, 'rmse_3d']:.6f}",
        },
        "median": {
            "file": str(median_idx),
            "rmse_3d": f"{results_df.loc[median_idx, 'rmse_3d']:.6f}",
        },
        "worst": {
            "file": str(worst_idx),
            "rmse_3d": f"{results_df.loc[worst_idx, 'rmse_3d']:.6f}",
        },
    }


def make_output_dirs(root: Path, timestamp: Optional[str] = None) -> Path:
    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_file").mkdir(exist_ok=True)
    return out


def _compute_e3_series(df: pd.DataFrame) -> np.ndarray:
    ex = df["x"].to_numpy() - df["gx"].to_numpy()
    ey = df["y"].to_numpy() - df["gy"].to_numpy()
    ez = df["z"].to_numpy() - df["gz"].to_numpy()
    return np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)


def plot_per_file_overlays(
    df: pd.DataFrame, file_key: str, out_dir: Path
) -> None:
    # Time index
    t = np.arange(len(df))
    ex = df["x"].to_numpy() - df["gx"].to_numpy()
    ey = df["y"].to_numpy() - df["gy"].to_numpy()
    ez = df["z"].to_numpy() - df["gz"].to_numpy()
    e3 = np.sqrt(ex ** 2 + ey ** 2 + ez ** 2)

    # Per-axis time series
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, df["x"], label="x")
    axes[0].plot(t, df["gx"], label="gx", alpha=0.7)
    axes[0].set_ylabel("X")
    axes[0].legend(loc="best")
    axes[1].plot(t, df["y"], label="y")
    axes[1].plot(t, df["gy"], label="gy", alpha=0.7)
    axes[1].set_ylabel("Y")
    axes[1].legend(loc="best")
    axes[2].plot(t, df["z"], label="z")
    axes[2].plot(t, df["gz"], label="gz", alpha=0.7)
    axes[2].set_ylabel("Z")
    axes[2].legend(loc="best")
    axes[3].plot(t, e3, color="tab:red", label="|e| (3D)")
    axes[3].set_ylabel("3D error")
    axes[3].set_xlabel("Sample")
    axes[3].legend(loc="best")
    fig.suptitle(f"Time series and 3D error: {file_key}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "per_file" / f"{file_key}_timeseries.png", dpi=150)
    plt.close(fig)

    # 3D trajectory overlay
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["gx"], df["gy"], df["gz"], label="ref", alpha=0.8)
    ax.plot(df["x"], df["y"], df["z"], label="meas", alpha=0.8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    ax.set_title(f"3D Trajectory: {file_key}")
    fig.tight_layout()
    fig.savefig(out_dir / "per_file" / f"{file_key}_3dtraj.png", dpi=150)
    plt.close(fig)

    # Error distribution per axis
    err_df = pd.DataFrame({"ex": ex, "ey": ey, "ez": ez})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=np.abs(err_df), ax=ax)
    ax.set_title(f"Absolute error distribution per axis: {file_key}")
    ax.set_ylabel("|error|")
    fig.tight_layout()
    fig.savefig(
        out_dir / "per_file" / f"{file_key}_abs_error_box.png", dpi=150
    )
    plt.close(fig)


def plot_aggregate(results_df: pd.DataFrame, out_dir: Path) -> None:
    # Bar chart for RMSE by axis and 3D
    rmse_cols = ["x_rmse", "y_rmse", "z_rmse", "rmse_3d"]
    fig, ax = plt.subplots(figsize=(12, 5))
    results_df[rmse_cols].plot(kind="bar", ax=ax)
    ax.set_title("RMSE per file (axis and 3D)")
    ax.set_xlabel("File")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_per_file.png", dpi=150)
    plt.close(fig)

    # Scatter: MAE vs RMSE (3D)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        results_df["mae_3d"],
        results_df["rmse_3d"],
        c=np.arange(len(results_df)),
        cmap="viridis",
    )
    for idx in results_df.index:
        ax.annotate(
            str(idx),
            (results_df.loc[idx, "mae_3d"], results_df.loc[idx, "rmse_3d"]),
            fontsize=8,
            alpha=0.7,
        )
    ax.set_xlabel("MAE 3D")
    ax.set_ylabel("RMSE 3D")
    ax.set_title("MAE vs RMSE (3D) by file")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_vs_rmse_3d.png", dpi=150)
    plt.close(fig)


def analyze_dataset(
    dataset_name: str,
    timestamp: Optional[str] = None,
    collect_errors: bool = True,
) -> Tuple[Path, pd.DataFrame, Optional[np.ndarray]]:
    data_dir = BASE_DATA_DIR / dataset_name
    reports_root = data_dir / "reports"
    print(f"[INFO] Data directory ({dataset_name}): {data_dir}")
    datasets = load_all_csvs(data_dir)
    print(
        f"[INFO] Loaded {len(datasets)} datasets: {list(datasets.keys())}"
    )

    out_dir = make_output_dirs(reports_root, timestamp)
    print(f"[INFO] Reports will be saved to: {out_dir}")

    # Compute per-file metrics
    rows = []
    e3_all: List[np.ndarray] = []
    for key, df in datasets.items():
        m = compute_metrics_for_df(df)
        m["file"] = key
        rows.append(m)
        if collect_errors:
            e3_all.append(_compute_e3_series(df))

    results_df = pd.DataFrame(rows).set_index("file").sort_values("rmse_3d")

    # Save metrics
    results_csv = out_dir / "metrics_summary.csv"
    results_df.to_csv(results_csv, float_format="%.6f")

    # Print a concise table to console
    display_cols = [
        "n_samples",
        "rmse_3d",
        "mae_3d",
        "median_3d",
        "p95_3d",
        "max_3d",
        "x_rmse",
        "y_rmse",
        "z_rmse",
        "x_mae",
        "y_mae",
        "z_mae",
        "x_bias",
        "y_bias",
        "z_bias",
    ]
    print("\n=== Metrics Summary (sorted by RMSE 3D) ===")
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        160,
    ):
        print(results_df[display_cols])

    # Summarize best/median/worst
    summary = summarize_across_files(results_df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[INFO] Best/Median/Worst by RMSE_3D:")
    print(json.dumps(summary, indent=2))

    # Generate per-file plots
    for key, df in datasets.items():
        plot_per_file_overlays(df, key, out_dir)

    # Aggregate plots
    plot_aggregate(results_df, out_dir)

    # Also generate focused plots for best/median/worst
    for label, entry in summary.items():
        key = entry["file"]
        plot_per_file_overlays(datasets[key], f"{key}_{label}", out_dir)

    print(f"\n[INFO] Done. Results saved to: {out_dir}")

    e3_concat = np.concatenate(e3_all) if (collect_errors and e3_all) else None
    # Optional: dataset-level CDF plot of e3
    if e3_concat is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.sort(e3_concat)
        y = np.linspace(0, 1, len(x), endpoint=False)
        ax.plot(x, y, label=f"{dataset_name} |e| CDF")
        ax.set_xlabel("|error| (3D)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Dataset-level CDF of 3D error: {dataset_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cdf_e3.png", dpi=150)
        plt.close(fig)

    return out_dir, results_df, e3_concat


def _bootstrap_ci_mean(
    delta: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(delta)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(np.mean(delta[idx])))
    deltas = np.asarray(deltas)
    mean_est = float(np.mean(deltas))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return mean_est, float(lo), float(hi)


def compare_datasets(
    name_a: str,
    df_a: pd.DataFrame,
    e3_a: Optional[np.ndarray],
    name_b: str,
    df_b: pd.DataFrame,
    e3_b: Optional[np.ndarray],
    out_root: Path,
    timestamp: Optional[str] = None,
) -> Path:
    out_dir = make_output_dirs(out_root, timestamp)
    print(f"[INFO] Comparison report: {out_dir}")

    # Align files present in both
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError("No common files to compare between datasets")
    A = df_a.loc[common]
    B = df_b.loc[common]

    # Compute per-file deltas for key metrics (B - A):
    # negative means B better if lower is better
    metrics = ["rmse_3d", "x_rmse", "y_rmse", "z_rmse", "mae_3d", "p95_3d"]
    deltas = {}
    for m in metrics:
        deltas[m] = (B[m] - A[m]).astype(float)

    # Summary stats and bootstrap CI for mean delta
    summary = {}
    for m in metrics:
        d = deltas[m].to_numpy()
        mean_diff = float(np.mean(d))
        median_diff = float(np.median(d))
        # fraction where B < A (B better if lower)
        win_rate = float(np.mean(d < 0))
        mean_est, lo, hi = _bootstrap_ci_mean(d)
        # Cohen's d for paired deltas
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
        effect = mean_diff / sd if (sd and sd != 0) else np.nan
        summary[m] = {
            "mean_delta": mean_diff,
            "median_delta": median_diff,
            "win_rate_B_lt_A": win_rate,
            "boot_mean": mean_est,
            "boot_ci95": [lo, hi],
            "cohens_d": effect,
        }

    # Save JSON summary
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "A": name_a,
                "B": name_b,
                "summary": summary,
                "n_files": int(len(common)),
            },
            indent=2,
        )
    )

    # Scatter B vs A (rmse_3d)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        A["rmse_3d"], B["rmse_3d"], c=np.arange(len(common)), cmap="viridis"
    )
    lims = [
        float(np.min([A["rmse_3d"].min(), B["rmse_3d"].min()])),
        float(np.max([A["rmse_3d"].max(), B["rmse_3d"].max()])),
    ]
    pad = 0.05 * (lims[1] - lims[0])
    ax.plot(
        [lims[0] - pad, lims[1] + pad],
        [lims[0] - pad, lims[1] + pad],
        "k--",
        alpha=0.5,
        label="y=x",
    )
    for idx in common:
        ax.annotate(
            str(idx),
            (A.loc[idx, "rmse_3d"], B.loc[idx, "rmse_3d"]),
            fontsize=8,
            alpha=0.7,
        )
    ax.set_xlabel(f"{name_a} RMSE 3D")
    ax.set_ylabel(f"{name_b} RMSE 3D")
    ax.set_title(f"Per-file RMSE 3D: {name_b} vs {name_a}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_rmse3d_B_vs_A.png", dpi=150)
    plt.close(fig)

    # Delta bar plot (rmse_3d)
    fig, ax = plt.subplots(figsize=(12, 5))
    (B["rmse_3d"] - A["rmse_3d"]).plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_title(f"Delta RMSE 3D (B - A): {name_b} - {name_a}")
    ax.set_ylabel("Delta RMSE 3D (negative favors B)")
    ax.set_xlabel("File")
    fig.tight_layout()
    fig.savefig(out_dir / "bar_delta_rmse3d.png", dpi=150)
    plt.close(fig)

    # Boxplot comparison across files for per-axis RMSE
    comp_df = pd.DataFrame({
        f"{name_a}_x_rmse": A["x_rmse"], f"{name_b}_x_rmse": B["x_rmse"],
        f"{name_a}_y_rmse": A["y_rmse"], f"{name_b}_y_rmse": B["y_rmse"],
        f"{name_a}_z_rmse": A["z_rmse"], f"{name_b}_z_rmse": B["z_rmse"],
    })
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=comp_df, ax=ax)
    ax.set_title(f"Per-axis RMSE across files: {name_a} vs {name_b}")
    ax.set_ylabel("RMSE")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "box_axes_rmse.png", dpi=150)
    plt.close(fig)

    # CDF overlay of 3D error across all samples (if collected)
    if (e3_a is not None) and (e3_b is not None):
        fig, ax = plt.subplots(figsize=(8, 6))
        for arr, label in [(e3_a, name_a), (e3_b, name_b)]:
            x = np.sort(arr)
            y = np.linspace(0, 1, len(x), endpoint=False)
            ax.plot(x, y, label=label)
        ax.set_xlabel("|error| (3D)")
        ax.set_ylabel("CDF")
        ax.set_title("CDF of 3D error (all samples)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "cdf_compare_e3.png", dpi=150)
        plt.close(fig)

    # Bland-Altman for rmse_3d across files
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_vals = 0.5 * (A["rmse_3d"] + B["rmse_3d"]).to_numpy()
    diff_vals = (B["rmse_3d"] - A["rmse_3d"]).to_numpy()
    ax.scatter(mean_vals, diff_vals, s=30, alpha=0.8)
    ax.axhline(
        np.mean(diff_vals), color="tab:red", linestyle="--", label="mean diff"
    )
    sd = np.std(diff_vals, ddof=1) if len(diff_vals) > 1 else 0.0
    ax.axhline(
        np.mean(diff_vals) + 1.96 * sd,
        color="gray",
        linestyle=":",
        label="±1.96 SD",
    )
    ax.axhline(np.mean(diff_vals) - 1.96 * sd, color="gray", linestyle=":")
    ax.set_xlabel("Mean RMSE 3D")
    ax.set_ylabel("Difference (B - A)")
    ax.set_title(f"Bland-Altman: {name_b} vs {name_a} (RMSE 3D)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "bland_altman_rmse3d.png", dpi=150)
    plt.close(fig)

    # Markdown human-readable summary
    lines = [
        f"# Comparison: {name_b} vs {name_a}",
        f"Common files: {len(common)}",
        "",
        "## Key metric: RMSE 3D",
    ]
    s = summary["rmse_3d"]
    lines += [
        f"Mean delta (B - A): {s['mean_delta']:.6f}",
        (
            "Bootstrap mean 95% CI: "
            f"[{s['boot_ci95'][0]:.6f}, {s['boot_ci95'][1]:.6f}]"
        ),
        f"Win rate (B < A): {s['win_rate_B_lt_A']*100:.1f}%",
        f"Effect size (Cohen's d): {s['cohens_d']:.3f}",
        "",
        (
            "Interpretation: Negative mean delta and CI below 0 favor "
            "B (lower error)."
        ),
    ]
    (out_dir / "comparison_summary.md").write_text("\n".join(lines))

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze flight CSVs under new_data/<dataset>. "
            "Provide one or more dataset names (e.g., rvolmea piddob)."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Datasets to analyze (choose from: rvolmea, piddob). "
            "Default: both"
        ),
    )
    args = parser.parse_args()

    dsets = args.datasets or ["rvolmea", "piddob"]
    allowed = {"rvolmea", "piddob"}
    for ds in dsets:
        if ds not in allowed:
            raise SystemExit(
                f"Invalid dataset '{ds}'. Allowed: {sorted(allowed)}"
            )
    # Use a common timestamp if running multiple datasets in one invocation
    common_ts = (
        dt.datetime.now().strftime("%Y%m%d_%H%M%S") if len(dsets) > 1 else None
    )

    results_map = {}
    for name in dsets:
        out_dir, df_metrics, e3_concat = analyze_dataset(name, common_ts)
        results_map[name] = {
            "out_dir": out_dir,
            "metrics": df_metrics,
            "e3": e3_concat,
        }

    print("\n[INFO] Completed datasets:")
    for name, obj in results_map.items():
        print(f"  - {name}: {obj['out_dir']}")

    # If both rvolmea and piddob are present, run comparison
    if {"rvolmea", "piddob"}.issubset(results_map.keys()):
        compare_root = BASE_DATA_DIR / "reports_compare"
        compare_out = compare_datasets(
            "rvolmea",
            results_map["rvolmea"]["metrics"],
            results_map["rvolmea"]["e3"],
            "piddob",
            results_map["piddob"]["metrics"],
            results_map["piddob"]["e3"],
            compare_root,
            common_ts,
        )
        print(f"\n[INFO] Comparison report: {compare_out}")


if __name__ == "__main__":
    main()
