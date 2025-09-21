#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sudoku_hyperparameters.py

Usage:
  python sudoku_hyperparameters.py <path_to_experimental_csv>

What it does:
- Reads the given experimental data CSV.
- Detects missingness type (MCAR/MAR/MNAR) from the file path/name.
- Creates output dir at script location: analysis/hyperparameter_tables/<missingness_type_lower>.
- Sets up a grid of hyperparameters (compact, fast Tier-0 grid).
- For each combo:
    * Computes p85 threshold (if requested) from the partner matrix built on the input df and (alpha,beta).
    * Calls SudokuImputer (iterative_graph_imputation_final) with LinearRegression base model and max_rounds=1.
    * Writes the imputed dataframe to {UID}_imputed.csv
    * Writes a {UID}_runtime.txt file with "<seconds> seconds"
- Writes a metadata CSV mapping UID -> hyperparameters & file paths.
- Minimizes memory footprint: each run writes to disk and releases references before moving on.
"""

import argparse
import gc
import os
import re
import time
from itertools import product
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---- Import your imputer and helpers from graph_imputer.py ----
# Assumes graph_imputer.py is in the SAME directory as this script.
SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from graph_imputer import (  # type: ignore
    iterative_graph_imputation_final,
    compute_biavailability_matrix,
    compute_spearman_matrix,
)

# ------------------------- Helpers -------------------------

MISSINGNESS_PATTERN = re.compile(r"(MCAR|MNAR|MAR)", re.IGNORECASE)


def detect_missingness_type(path_str: str) -> str:
    """Infer missingness type from path or filename (MCAR/MAR/MNAR)."""
    match = MISSINGNESS_PATTERN.search(path_str)
    if not match:
        raise ValueError(
            "Could not detect missingness type (MCAR/MAR/MNAR) in the provided path."
        )
    return match.group(1).upper()


def make_output_dir(base_dir: Path, missingness_type: str) -> Path:
    """Create analysis/hyperparameter_tables/<mtype_lower> under script directory."""
    out_dir = base_dir / "analysis" / "hyperparameter_tables" / missingness_type.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_partner_matrix(df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    """Build partner matrix = (Spearman ** alpha) * (Bi-availability ** beta)."""
    biavail = compute_biavailability_matrix(df)
    assoc = compute_spearman_matrix(df)
    with np.errstate(invalid="ignore"):
        partner_matrix = (assoc ** alpha) * (biavail ** beta)
    return partner_matrix


def percentile_threshold_from_partner_matrix(
    partner_matrix: pd.DataFrame, percentile: float
) -> float:
    """Compute an off-diagonal percentile of the partner matrix."""
    arr = partner_matrix.to_numpy(dtype=float)
    # Mask diagonal and NaNs
    n = arr.shape[0]
    mask = ~np.eye(n, dtype=bool) & np.isfinite(arr)
    values = arr[mask]
    if values.size == 0:
        # Degenerate case; fallback to a very low threshold to retain edges
        return float("-inf")
    return float(np.percentile(values, percentile))


def generate_grid() -> List[Dict[str, Any]]:
    """
    Grid per user spec:

    * method: ["degree", "betweenness", "eigenvector"]
    * prioritization_mode: ["centrality", "utility"]
    * partner_proportion: [0.08, 0.15, 0.25]
    * edge_threshold_rule: [None, "p85"]
    * (alpha, beta): [(1.0,1.0), (0.5,1.5), (1.5,0.5)]
    * model_func: [LinearRegression]
    * max_rounds: [1]
    """
    methods = ["degree", "betweenness", "eigenvector"]
    ordering = ["centrality", "utility"]
    partner_props = [0.08, 0.15, 0.25]
    edge_rules = [None, "p85"]
    alpha_beta_pairs: List[Tuple[float, float]] = [(1.0, 1.0), (0.5, 1.5), (1.5, 0.5)]
    model_funcs = [LinearRegression]  # speed
    max_rounds = [1]  # speed

    grid = []
    for m, o, pp, er, (a, b), mf, mr in product(
        methods, ordering, partner_props, edge_rules, alpha_beta_pairs, model_funcs, max_rounds
    ):
        grid.append(
            dict(
                method=m,
                prioritization_mode=o,
                partner_proportion=pp,
                edge_threshold_rule=er,
                alpha=a,
                beta=b,
                model_func_name="LinearRegression",
                max_rounds=mr,
            )
        )
    return grid


def make_uid(idx: int) -> str:
    """Deterministic, compact unique identifier for each configuration."""
    return f"HP{idx:04d}"


# ------------------------- Main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SudokuImputer hyperparameter grid.")
    parser.add_argument("csv_path", type=str, help="Path to experimental data CSV")
    args = parser.parse_args()

    input_csv = Path(args.csv_path).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Detect missingness from path or file name
    missingness_type = detect_missingness_type(str(input_csv))

    # Prepare output directory at script location
    out_dir = make_output_dir(SCRIPT_DIR, missingness_type)
    print(f"[INFO] Output directory: {out_dir}")

    # Load input dataframe once; copy per-run to minimize I/O
    print(f"[INFO] Reading input CSV: {input_csv}")
    df_in = pd.read_csv(input_csv)

    # Build the full grid
    grid = generate_grid()
    print(f"[INFO] Grid size: {len(grid)} configurations")

    # Prepare metadata rows
    metadata_rows: List[Dict[str, Any]] = []

    # Run each config
    for idx, params in enumerate(grid, start=1):
        uid = make_uid(idx)

        # Compute numeric edge threshold if rule is "p85"
        edge_rule = params["edge_threshold_rule"]
        if edge_rule == "p85":
            # Build partner matrix based on (alpha, beta) for this config
            pm = build_partner_matrix(df_in, alpha=params["alpha"], beta=params["beta"])
            edge_threshold_value = percentile_threshold_from_partner_matrix(pm, 85.0)
            # Release partner matrix early
            del pm
        else:
            edge_threshold_value = None

        # File outputs
        imputed_csv_path = out_dir / f"{uid}_imputed.csv"
        runtime_txt_path = out_dir / f"{uid}_runtime.txt"

        # Record metadata entry (paths will be valid after write)
        meta_entry = {
            "uid": uid,
            "input_csv": str(input_csv),
            "missingness_type": missingness_type,
            "method": params["method"],
            "prioritization_mode": params["prioritization_mode"],
            "partner_proportion": params["partner_proportion"],
            "edge_threshold_rule": edge_rule if edge_rule is not None else "None",
            "edge_threshold_value": (
                float(edge_threshold_value) if edge_threshold_value is not None else ""
            ),
            "alpha": params["alpha"],
            "beta": params["beta"],
            "model_func": params["model_func_name"],
            "max_rounds": params["max_rounds"],
            "output_imputed_csv": str(imputed_csv_path),
            "output_runtime_txt": str(runtime_txt_path),
        }

        # Prepare call to SudokuImputer
        imputer_kwargs = dict(
            method=params["method"],
            edge_threshold=edge_threshold_value,
            partner_proportion=params["partner_proportion"],
            strategy="graph_aware_MLR",  # per your implementation
            prioritization_mode=params["prioritization_mode"],
            alpha=params["alpha"],
            beta=params["beta"],
            model_func=LinearRegression,  # speed
            max_rounds=params["max_rounds"],
        )

        # Run and time
        start = time.perf_counter()
        try:
            df_out, logs = iterative_graph_imputation_final(df_in.copy(), **imputer_kwargs)
        except Exception as e:
            # If a configuration fails, record it and continue
            meta_entry["status"] = "failed"
            meta_entry["error"] = repr(e)
            metadata_rows.append(meta_entry)
            print(f"[WARN] {uid} failed: {e}")
            continue
        end = time.perf_counter()
        elapsed = end - start

        # Write outputs
        # 1) Imputed CSV
        df_out.to_csv(imputed_csv_path, index=False)

        # 2) Runtime TXT
        with open(runtime_txt_path, "w", encoding="utf-8") as f:
            f.write(f"{elapsed:.4f} seconds")

        meta_entry["status"] = "ok"
        meta_entry["error"] = ""
        meta_entry["runtime_seconds"] = f"{elapsed:.4f}"

        # Append to metadata table
        metadata_rows.append(meta_entry)

        # ---- Explicitly free memory before next iteration ----
        del df_out, logs
        gc.collect()

        print(f"[OK] {uid}  ({elapsed:.2f}s)")

    # Write metadata CSV at the end
    metadata_csv_path = out_dir / "metadata.csv"
    meta_df = pd.DataFrame(metadata_rows)
    # Order columns for readability
    col_order = [
        "uid",
        "status",
        "error",
        "runtime_seconds",
        "input_csv",
        "missingness_type",
        "method",
        "prioritization_mode",
        "partner_proportion",
        "edge_threshold_rule",
        "edge_threshold_value",
        "alpha",
        "beta",
        "model_func",
        "max_rounds",
        "output_imputed_csv",
        "output_runtime_txt",
    ]
    meta_df = meta_df[[c for c in col_order if c in meta_df.columns]]
    meta_df.to_csv(metadata_csv_path, index=False)
    print(f"[INFO] Wrote metadata: {metadata_csv_path}")

    print("[DONE] All runs complete.")


if __name__ == "__main__":
    main()
