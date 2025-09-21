#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure we can import graph_imputer.py from the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from graph_imputer import iterative_graph_imputation_final  # noqa: E402


def _json_safe(obj):
    """Recursively convert NumPy/pandas types to vanilla Python for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val):
            return None
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if obj is pd.NaT:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    try:
        json.dumps(obj)  # type: ignore[name-defined]
        return obj
    except Exception:
        return str(obj)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run iterative_graph_imputation_final() on a CSV."
    )
    p.add_argument(
        "input_csv",
        type=str,
        help=("Path like "
              "datasets/experimental_data/numerical/[DATASET]/[MISSINGNESS_TYPE]/"
              "[dataset]_[missingness_type]_[missingness_proportion].csv"),
    )
    # Optional knobs (forwarded to the imputer)
    p.add_argument("--method", type=str, default="degree",
                   choices=["degree", "betweenness", "eigenvector"],
                   help="Centrality method.")
    p.add_argument("--edge-threshold", type=float, default=None,
                   help="Min edge weight to include in the graph.")
    p.add_argument("--partner-proportion", type=float, default=0.2,
                   help="Proportion of partners to use (0–1].")
    p.add_argument("--strategy", type=str, default="graph_aware_MLR",
                   choices=["graph_aware_MLR", "MICE_MLR"],
                   help="Imputation strategy.")
    p.add_argument("--prioritization", type=str, default="centrality",
                   choices=["centrality", "utility"],
                   help="Feature selection mode per round.")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Association (Spearman) exponent.")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Bi-availability exponent.")
    p.add_argument("--max-rounds", type=int, default=3,
                   help="Maximum imputation rounds.")
    return p.parse_args()


def derive_output_paths(in_path: Path):
    """
    Input:
      datasets/experimental_data/numerical/[DATASET]/[MISSINGNESS_TYPE]/
      [dataset]_[missingness_type]_[missingness_proportion].csv

    Output dir:
      outputs/[missingness_type_lower]/[dataset_lower]/

    Filenames:
      imputed_[prop].csv
      execution_time_[prop].txt
      logs_[prop].json
    """
    # Grab components relative to ".../numerical"
    parts = in_path.parts
    try:
        idx = parts.index("numerical")
    except ValueError:
        raise ValueError(
            "Input path must contain 'numerical' segment per the required structure."
        )
    try:
        dataset = parts[idx + 1]
        missingness_type = parts[idx + 2]
    except IndexError:
        raise ValueError(
            "Could not parse [DATASET]/[MISSINGNESS_TYPE] from the input path."
        )

    # Extract missingness proportion from the filename stem
    stem = in_path.stem  # e.g., eeg_MCAR_0.2
    pieces = stem.split("_")
    if len(pieces) < 3:
        raise ValueError(
            "Filename must be [dataset]_[missingness_type]_[missingness_proportion].csv"
        )
    prop = pieces[-1]  # "0.2" from "..._0.2"

    out_dir = Path("outputs") / missingness_type.lower() / dataset.lower()
    out_csv = out_dir / f"imputed_{prop}.csv"
    time_txt = out_dir / f"execution_time_{prop}.txt"
    logs_json = out_dir / f"logs_{prop}.json"

    return out_dir, out_csv, time_txt, logs_json, dataset, missingness_type, prop


def main():
    args = parse_args()

    in_path = Path(args.input_csv).resolve()
    if not in_path.exists():
        sys.stderr.write(f"[error] Input file not found: {in_path}\n")
        sys.exit(1)

    out_dir, out_csv_path, time_txt_path, logs_json_path, dataset, mtype, prop = derive_output_paths(in_path)

    # Make sure destination exists (safe even if it already exists)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Reading CSV: {in_path}")
    df_in = pd.read_csv(in_path)

    total_missing_before = int(df_in.isna().sum().sum())
    print(f"[info] Shape: {df_in.shape[0]} rows x {df_in.shape[1]} cols")
    print(f"[info] Total missing values before: {total_missing_before}")
    print(f"[info] Output directory: {out_dir}")

    # Run and time the imputation
    print("[info] Running iterative_graph_imputation_final() ...")
    t0 = time.perf_counter()
    df_out, logs = iterative_graph_imputation_final(
        df_in,
        method=args.method,
        edge_threshold=args.edge_threshold,
        partner_proportion=args.partner_proportion,
        strategy=args.strategy,
        prioritization_mode=args.prioritization,
        alpha=args.alpha,
        beta=args.beta,
        max_rounds=args.max_rounds,
    )
    elapsed = time.perf_counter() - t0

    total_missing_after = int(df_out.isna().sum().sum())

    # Save outputs
    df_out.to_csv(out_csv_path, index=False)
    with open(time_txt_path, "w") as f:
        f.write(f"{elapsed:.4f} seconds")

    safe_logs = _json_safe(logs)
    with open(logs_json_path, "w") as f:
        json.dump(safe_logs, f, indent=2, allow_nan=False)

    print(f"[ok] Saved imputed CSV         -> {out_csv_path}")
    print(f"[ok] Saved execution time      -> {time_txt_path}  ({elapsed:.4f} seconds)")
    print(f"[ok] Saved logs JSON           -> {logs_json_path}")
    print(f"[summary] Missing values: before={total_missing_before}  after={total_missing_after}")

    if total_missing_after > 0:
        print("[note] Some missing values remain. Consider adjusting --edge-threshold, "
              "--partner-proportion, or --max-rounds.")


if __name__ == "__main__":
    main()
