# 🧩 SudokuImputer 🧩

SudokuImputer is a graph-based imputation algorithm designed for numerical datasets with synthetic or real-world missingness. Inspired by strategies in Sudoku solving, the method prioritizes features to impute based on centrality and "imputability" scores, enabling robust performance across MCAR, MAR, and MNAR scenarios.

---

## Features
- Handles datasets with **10–100+ features**.
- Supports **MCAR, MAR, MNAR** missingness types.
- Prioritizes features using **graph-based centrality** measures.
- Imputation guided by **utility-based ordering** (imputability ratio).
- Benchmarked against classical and modern imputation methods ("SudokuImputer: A Best-in-Class Graph Framework for MNAR and MAR Missing Data Imputation" submitted for publication in the Journal of Big Data and Artificial Intelligence, Vol. 3.)
- Outputs:
  - Imputed datasets (`.csv`)
  - Evaluation logs and metrics (`RMSE`, `R²`)
  - Execution times

---
## Installation

To use from command line:
```bash
git clone https://github.com/<username>/SudokuImputer.git
cd SudokuImputer
pip install -r requirements.txt
```

or, to use in Jupyter Notebook environment:
```bash
from graph_imputer import iterative_graph_imputation_final
imputed_dataset = iterative_graph_imputation_final(experimental_dataset, [hyperparams]...)
```
(Please see Demo Implementation for an example)

---
## Repository Structure

SudokuImputer/
├── Manuscript_Analysis_Pipeline/            # Benchmark scripts & analysis
│   ├── analysis/
│   ├── datasets/
│   ├── outputs/
│   ├── 1_numerical_imputation_pipeline.py
│   └── 2_execute_sudoku.py
├── graph_imputer.py                         # Core algorithm
├── Demo Implementation.ipynb                # Sample implementation of SudokuImputer
├── LICENSE.txt
└── README.md

---
## Algorithmic Design and Functionality

SudokuImputer frames missing value imputation as a structured sequence of "puzzle solving" steps:

1. **Graph Construction**  
   - Each feature is represented as a node in a graph.  
   - Edges encode statistical associations (e.g., correlation, mutual information) between features.  
   - Edge weights reflect predictive utility, forming a dependency network.

2. **Centrality-Based Prioritization**  
   - Features are ranked by centrality measures (e.g., betweenness, degree).  
   - This determines which features are most informative for propagating imputations.  

3. **Imputability Ratio**  
   - A dynamic score that quantifies the ease of imputing a feature based on observed-to-missing proportions and available predictors.  
   - Serves as a "Sudoku-like" rule: fill what is most solvable first.  

4. **Iterative Imputation**  
   - Features are imputed sequentially following the prioritized order.  
   - At each step, regressors are trained on observed data, predictions are made for missing entries, and the graph is updated.  

5. **Convergence and Output**  
   - The process continues until all missing values are imputed.  
   - Outputs include the completed dataset, execution logs, and benchmarking metrics.  

This design ensures that highly informative features are imputed early, reducing noise propagation and improving robustness under MCAR, MAR, and MNAR missingness.

---
## Citation

@misc{<your_citation_here>,
  title  = {SudokuImputer: Graph-based Feature Prioritization for Data Imputation},
  author = {<Your Name>},
  year   = {2025},
  url    = {https://github.com/<username>/SudokuImputer}
}

---
## Contributing

Contributions are welcome! Please open an issue, submit a pull request, or contact the corresponding author at vivek.kanpa@icahn.mssm.edu

---
