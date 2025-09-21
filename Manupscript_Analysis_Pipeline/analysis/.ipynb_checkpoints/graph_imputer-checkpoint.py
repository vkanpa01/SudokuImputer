# graph_imputer.py  — patched (MNAR gridsearch hardening)
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from typing import Callable, Optional, Tuple, Literal

# Sklearn: robustify base model training against partial missingness
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor


# ------------------------- Core helpers -------------------------

def compute_biavailability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise fraction of rows where BOTH columns are observed.
    """
    n = df.shape[1]
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            both = (~df.iloc[:, i].isna()) & (~df.iloc[:, j].isna())
            val = float(both.mean())
            matrix[i, j] = matrix[j, i] = val
    return pd.DataFrame(matrix, index=df.columns, columns=df.columns)


def compute_spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Spearman r; uses only rows observed for BOTH columns.
    """
    n = df.shape[1]
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                r = 1.0
            else:
                mask = (~df.iloc[:, i].isna()) & (~df.iloc[:, j].isna())
                if mask.sum() > 2:
                    col_i = df.loc[mask, df.columns[i]]
                    col_j = df.loc[mask, df.columns[j]]
                    r, _ = spearmanr(col_i, col_j)
                    if not np.isfinite(r):
                        r = 0.0
                else:
                    r = 0.0
            matrix[i, j] = matrix[j, i] = float(r)
    return pd.DataFrame(matrix, index=df.columns, columns=df.columns)


def _is_finite_num(x) -> bool:
    try:
        return np.isfinite(x)
    except Exception:
        return False


def build_weighted_graph(matrix: pd.DataFrame, directed: bool = False, edge_threshold: Optional[float] = None) -> nx.Graph:
    """
    Build a (by default) UNDIRECTED weighted graph from a partner/association matrix.
    - Skips non-finite weights
    - Applies threshold only to finite weights
    """
    G = nx.DiGraph() if directed else nx.Graph()
    for i in matrix.index:
        for j in matrix.columns:
            if i == j:
                continue
            w = matrix.loc[i, j]
            if _is_finite_num(w):
                if (edge_threshold is None) or (w >= edge_threshold):
                    # Store as regular float to avoid numpy scalar surprises
                    G.add_edge(i, j, weight=float(w))
    return G


def compute_node_centrality(G: nx.Graph, method: str = "degree") -> pd.Series:
    """
    Centrality with stable eigenvector handling:
    - Always computes on an UNDIRECTED view for eigenvector
    - Raises max_iter, tight tol
    - Falls back to degree on convergence issues
    """
    if method == "degree":
        cent = nx.degree_centrality(G)
    elif method == "betweenness":
        cent = nx.betweenness_centrality(G, weight="weight")
    elif method == "eigenvector":
        H = G.to_undirected()
        try:
            cent = nx.eigenvector_centrality(H, max_iter=2000, tol=1e-06, weight="weight")
        except Exception:
            cent = nx.degree_centrality(H)
    else:
        raise ValueError("Unknown centrality method.")
    return pd.Series(cent).sort_values(ascending=False)


# ------------------------- Partner selection + utilities -------------------------

def get_top_partner_nodes(c_node: str, partner_node_matrix: pd.DataFrame, proportion: float = 0.2):
    """
    Choose top-k partners by weight for candidate node.
    """
    n_partners = max(1, int(partner_node_matrix.shape[1] * float(proportion)))
    return (
        partner_node_matrix.loc[c_node]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_values(ascending=False)
        .head(n_partners)
        .index
        .tolist()
    )


def compute_imputability(df: pd.DataFrame, c_node: str, partners: list) -> float:
    """
    Fraction of missing rows in c_node that have at least one available partner.
    """
    mask_missing = df[c_node].isna()
    if mask_missing.sum() == 0:
        return 0.0
    mask_available = df[partners].notna().any(axis=1)
    return float((mask_missing & mask_available).sum() / mask_missing.sum())


def get_most_central_node_with_missing(df: pd.DataFrame, centrality: pd.Series, already_imputed: set):
    for node in centrality.index:
        if df[node].isna().any() and node not in already_imputed:
            return node
    return None


def _fit_predict_row(model_factory: Callable, X_train: pd.DataFrame, y_train: pd.Series, x_row: np.ndarray) -> float:
    """
    Fit a NA-robust model (SimpleImputer + model) and predict a single value.
    """
    # Always wrap user model in a median-imputing pipeline for robustness.
    model = make_pipeline(SimpleImputer(strategy="median"), model_factory())
    model.fit(X_train, y_train)
    pred = model.predict(x_row.reshape(1, -1))
    return float(pred[0])


def impute_with_mlr(
    df: pd.DataFrame,
    c_node: str,
    partner_nodes: list,
    strategy: str,
    model_func: Callable = HistGradientBoostingRegressor,
):
    """
    Graph-aware MLR imputation:
    - Row-wise dynamic partner availability
    - NA-robust training via SimpleImputer in a pipeline
    - Median fallback for any still-missing rows in target column
    """
    df = df.copy()
    stats = {
        "partner_nodes": partner_nodes,
        "excluded_partners": [],
        "median_imputed_partners": [],
        "median_fallback_count": 0,
        "median_value": None,
        "total_missing": int(df[c_node].isna().sum()),
        "model_predictions": 0,
        "model_training_attempts": 0,
    }

    # Discard high-missing partners (>50% missing)
    usable_partners = [p for p in partner_nodes if df[p].isna().mean() <= 0.5]

    # If no usable partners, median-impute the whole column and bail
    if not usable_partners:
        val = float(df[c_node].median())
        df[c_node].fillna(val, inplace=True)
        stats["median_fallback_count"] = int(df[c_node].isna().sum())
        stats["median_value"] = val
        return df, stats

    if strategy == "graph_aware_MLR":
        model_cache = {}
        missing_idx = df[df[c_node].isna()].index

        for idx in missing_idx:
            row = df.loc[idx, usable_partners]
            available = row.dropna()
            if available.empty:
                continue

            key = tuple(sorted(available.index))
            if key not in model_cache:
                # Keep rows where target present; imputer will fix partial NaNs in features
                row_mask = df[c_node].notna()
                X_train = df.loc[row_mask, list(available.index)]
                y_train = df.loc[row_mask, c_node]

                if len(X_train) < 2:
                    # not enough data to fit a model — skip
                    continue

                try:
                    # Fit model with NA-robust pipeline
                    # (SimpleImputer handles any NaNs in X_train)
                    model_cache[key] = make_pipeline(SimpleImputer(strategy="median"), model_func())
                    model_cache[key].fit(X_train, y_train)
                    stats["model_training_attempts"] += 1
                except Exception:
                    # As a last resort, skip this available-set; prediction will be skipped
                    continue

            model = model_cache.get(key, None)
            if model is None:
                continue

            try:
                pred = model.predict(available.values.reshape(1, -1))
                df.at[idx, c_node] = float(pred[0])
                stats["model_predictions"] += 1
            except Exception:
                # Skip prediction failure; will be median-imputed later if still NaN
                continue

    # Median fallback for anything still missing
    fallback_mask = df[c_node].isna()
    if fallback_mask.any():
        val = float(df[c_node].median())
        df.loc[fallback_mask, c_node] = val
        stats["median_fallback_count"] = int(fallback_mask.sum())
        stats["median_value"] = val

    return df, stats


# ------------------------- Main iterative imputer -------------------------

def iterative_graph_imputation_final(
    df: pd.DataFrame,
    method: str = "degree",
    edge_threshold: Optional[float] = None,
    partner_proportion: float = 0.2,
    strategy: Literal["graph_aware_MLR", "MICE_MLR"] = "graph_aware_MLR",
    prioritization_mode: Literal["centrality", "utility"] = "centrality",
    alpha: float = 1.0,
    beta: float = 1.0,
    model_func: Callable = HistGradientBoostingRegressor,
    max_rounds: int = 3,
) -> Tuple[pd.DataFrame, dict]:
    """
    Iterative scheme:
    - Recompute partner weights each round using sign-preserving power:
          assoc_signed = sign(r) * |r|**alpha
          partner_matrix = assoc_signed * (biavail ** beta)
    - Build UNDIRECTED graph for centrality stability.
    - Keep fallbacks when centrality/utility produce no candidate.
    """
    df = df.copy()
    all_logs = {}

    for round_num in range(max_rounds):
        round_log = {}
        already_imputed = set()

        # --- Matrices ---
        biavail = compute_biavailability_matrix(df)
        assoc = compute_spearman_matrix(df)

        # Sign-preserving fractional power for robustness when alpha is fractional
        assoc_signed = np.sign(assoc.to_numpy(dtype=float)) * (np.abs(assoc.to_numpy(dtype=float)) ** float(alpha))
        partner_matrix = assoc_signed * (biavail.to_numpy(dtype=float) ** float(beta))
        partner_matrix = pd.DataFrame(partner_matrix, index=assoc.index, columns=assoc.columns)
        # Sanitize any residual non-finite weights
        partner_matrix = partner_matrix.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- Graph + centrality (undirected for stability) ---
        G = build_weighted_graph(partner_matrix, directed=False, edge_threshold=edge_threshold)
        if len(G) == 0:
            # Degenerate case: no edges survived thresholding — relax by ignoring threshold for centrality only
            G = build_weighted_graph(partner_matrix, directed=False, edge_threshold=None)

        centrality = compute_node_centrality(G, method=method)

        # --- Impute until no missing or we run out of reasonable candidates ---
        while df.isna().any().any():
            if prioritization_mode == "centrality":
                c_node = get_most_central_node_with_missing(df, centrality, already_imputed)
                if c_node is None:
                    # Fallback: pick column with most imputable missingness
                    candidates = [col for col in df.columns if df[col].isna().any() and col not in already_imputed]
                    if candidates:
                        best_col = None
                        best_score = -1.0
                        best_missing = -1
                        for col in candidates:
                            partners_fb = get_top_partner_nodes(col, partner_matrix, proportion=partner_proportion)
                            imp_score = compute_imputability(df, col, partners_fb)
                            miss_cnt = int(df[col].isna().sum())
                            if (imp_score > best_score) or (imp_score == best_score and miss_cnt > best_missing):
                                best_score = imp_score
                                best_missing = miss_cnt
                                best_col = col
                        c_node = best_col

            elif prioritization_mode == "utility":
                # On-demand import to avoid circulars (defined below)
                c_node = get_most_valuable_node_with_missing(df, partner_matrix, strategy, already_imputed, model_func)
            else:
                raise ValueError("Invalid prioritization mode.")

            if c_node is None:
                break

            partners = get_top_partner_nodes(c_node, partner_matrix, proportion=partner_proportion)
            imputability = compute_imputability(df, c_node, partners)

            df, stats = impute_with_mlr(df, c_node, partners, strategy, model_func)
            stats["selected_partners"] = partners
            stats["imputability_ratio"] = imputability
            round_log[c_node] = stats
            already_imputed.add(c_node)

        all_logs[f"round_{round_num + 1}"] = round_log

        # Early stop if complete
        if all(not df[col].isna().any() for col in df.columns):
            break

    return df, all_logs


# ------------------------- Utility-mode scoring -------------------------

def estimate_information_gain(
    df: pd.DataFrame,
    candidate_feature: str,
    partner_node_matrix: pd.DataFrame,
    strategy: str,
    model_func: Callable,
) -> float:
    """
    Proxy for utility: impute the candidate, then sum how many other columns become imputable.
    """
    top_partners = get_top_partner_nodes(candidate_feature, partner_node_matrix)
    df_temp, _ = impute_with_mlr(df.copy(), candidate_feature, top_partners, strategy, model_func)
    gain = 0.0
    for other in df.columns:
        if other != candidate_feature and df[other].isna().any():
            partners_other = get_top_partner_nodes(other, partner_node_matrix)
            gain += compute_imputability(df_temp, other, partners_other)
    return float(gain)


def get_most_valuable_node_with_missing(
    df: pd.DataFrame,
    partner_node_matrix: pd.DataFrame,
    strategy: str,
    already_imputed: set,
    model_func: Callable,
):
    candidates = [col for col in df.columns if df[col].isna().any() and col not in already_imputed]
    if not candidates:
        return None
    scores = {}
    for col in candidates:
        try:
            scores[col] = estimate_information_gain(df, col, partner_node_matrix, strategy, model_func)
        except Exception:
            scores[col] = -np.inf
    return max(scores, key=scores.get) if scores else None
