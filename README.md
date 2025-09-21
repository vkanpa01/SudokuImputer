# 🧩 SudokuImputer 🧩

SudokuImputer is a graph-based imputation algorithm designed for numerical datasets with synthetic or real-world missingness. Inspired by strategies in Sudoku solving, the method prioritizes features to impute based on centrality and "imputability" scores, enabling robust performance across MCAR, MAR, and MNAR scenarios.

---

## Features
- Handles datasets with **14–1000+ features**.
- Supports **MCAR, MAR, MNAR** missingness types.
- Prioritizes features using **graph-based centrality** measures.
- Imputation guided by **utility-based ordering** (imputability ratio).
- Benchmarked against classical and modern imputation methods.
- Outputs:
  - Imputed datasets (`.csv`)
  - Evaluation logs and metrics (`RMSE`, `R²`)
  - Execution times

---

## Installation

```bash
git clone https://github.com/<username>/SudokuImputer.git
cd SudokuImputer
pip install -r requirements.txt
