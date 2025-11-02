from datetime import datetime
from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flwr_datasets import FederatedDataset
from datasets import load_dataset 
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

# Pre-calculate categorical features and class imbalance from the full dataset once.
full_dataset = load_dataset("scikit-learn/adult-census-income", split="train")
df_full = full_dataset.to_pandas()
categorical_features_names = df_full.select_dtypes(include=['object']).columns.tolist()
if 'income' in categorical_features_names:
    categorical_features_names.remove('income')
ALL_CATEGORIES = { col: df_full[col].unique().tolist() for col in categorical_features_names }

# Registry for available data partitioners.
PARTITIONER_REGISTRY = {
    "uniform": lambda num_partitions, alpha: IidPartitioner(num_partitions=num_partitions),
    "dirichlet": lambda num_partitions, alpha: DirichletPartitioner(
        num_partitions=num_partitions, alpha=alpha, partition_by="income"
    ),
}

# Global cache for the FederatedDataset to avoid repeated downloads.
fds = None

# Instantiates the FederatedDataset with the specified partitioner.
def instantiate_fds(partitioner_type, num_partitions, alpha):
    global fds
    if fds is None:
        if partitioner_type not in PARTITIONER_REGISTRY:
            raise ValueError(f"Unknown partitioner: '{partitioner_type}'")
        partitioner_fn = PARTITIONER_REGISTRY[partitioner_type]
        partitioner = partitioner_fn(num_partitions, alpha)
        fds = FederatedDataset(
            dataset="scikit-learn/adult-census-income",
            partitioners={"train": partitioner},
        )
    return fds

# Loads a client's data partition, performs one-hot encoding, and splits it.
def load_data(partitioner_type, partition_id, num_partitions, test_fraction, seed, alpha):
    fds_ = instantiate_fds(partitioner_type, num_partitions, alpha)
    partition = fds_.load_partition(partition_id)
    df = partition.to_pandas()

    # Pre-process and encode the data partition.
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
    for col, all_cats in ALL_CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=all_cats)
    df_encoded = pd.get_dummies(df, columns=categorical_features_names, drop_first=True)

    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']

    # Split data into training and validation sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=seed)
    return xgb.DMatrix(X_train, label=y_train), xgb.DMatrix(X_test, label=y_test), len(X_train), len(X_test)

# Utility functions for key replacement and run directory creation.
def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        new_dict[new_key] = replace_keys(value, match, target) if isinstance(value, dict) else value
    return new_dict

def create_run_dir() -> tuple[Path, str]:
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / "outputs" / run_dir
    save_path.mkdir(parents=True, exist_ok=False)
    
    return save_path, run_dir