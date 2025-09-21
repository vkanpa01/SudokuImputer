
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from fancyimpute import SoftImpute, IterativeSVD
import pandas as pd
import numpy as np
import time
import os
import glob
import gc
from typing import Tuple


def impute_mean(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    imputer = SimpleImputer(strategy='mean')
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def impute_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def impute_mice(df: pd.DataFrame, max_iter: int = 10, seed: int = 42) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    imputer = IterativeImputer(max_iter=max_iter, random_state=seed)
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def impute_softimpute(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    imputer = SoftImpute()
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def impute_matrix_factorization(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    start_time = time.time()
    rank = max(1, min(df.shape) - 1)
    imputer = IterativeSVD(rank=rank)
    imputed = imputer.fit_transform(df)
    runtime = time.time() - start_time
    return pd.DataFrame(imputed, columns=df.columns), runtime


def get_imputation_methods():
    return {
        "mean": impute_mean,
        "median": impute_median,
        "mice": impute_mice,
        "knn": impute_knn,
        "softimpute": impute_softimpute,
        "matrix_factorization": impute_matrix_factorization
    }


def impute_all_numerical_datasets(input_dir='datasets/experimental_data/numerical', output_dir='datasets/imputed_data/numerical'):
    methods = get_imputation_methods()
    csv_files = glob.glob(f'{input_dir}/**/*.csv', recursive=True)

    for file_path in csv_files:
        if "ground_truth" in os.path.basename(file_path).lower():
            continue  # Skip ground truth files

        df = pd.read_csv(file_path)
        rel_path = os.path.relpath(file_path, input_dir)
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        out_file_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(out_file_dir, exist_ok=True)

        for method_name, impute_func in methods.items():
            imputed_file = os.path.join(out_file_dir, f"{dataset_name}_{method_name}_imputed.csv")
            runtime_file = os.path.join(out_file_dir, f"{dataset_name}_{method_name}_runtime.txt")

            # Skip if already done
            if os.path.exists(imputed_file) and os.path.exists(runtime_file):
                continue

            try:
                imputed_df, runtime = impute_func(df)
                imputed_df.to_csv(imputed_file, index=False)
                with open(runtime_file, 'w') as f:
                    f.write(f"{runtime:.4f}")
            except Exception as e:
                print(f"Failed to impute {file_path} with {method_name}: {e}")
            finally:
                del imputed_df
                gc.collect()

        del df
        gc.collect()


# Unit test
def test_imputation_methods():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 3, 4],
        'C': [1, np.nan, np.nan, 4]
    })

    methods = get_imputation_methods()
    for name, func in methods.items():
        try:
            result, runtime = func(df)
            assert isinstance(result, pd.DataFrame)
            assert not result.isnull().values.any()
            assert runtime >= 0
            print(f"{name} imputation test passed. Runtime: {runtime:.4f} seconds")
        except Exception as e:
            print(f"{name} imputation test failed: {e}")


if __name__ == "__main__":
    # test_imputation_methods()
    # Uncomment the line below to run on actual data
    impute_all_numerical_datasets()
