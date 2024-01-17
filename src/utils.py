from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def get_top_n_feature_importances(
    importances: List[float],
    features_names: List[str],
    top_n: int = -1
) -> Dict[str, float]:
    """
    Return most important features

    Parameters
    ----------
    importances : List[float]
        Importances
    features_names : List[str]
        Features names
    n : int, optional
        Number of most important features to return, by default -1.

    Returns
    -------
    Dict[str, float]
        Feature -> importance dict
    """

    res = dict()
    sort_idx = np.argsort(importances)
    if top_n == -1:
        top_n = len(importances)
    for idx in sort_idx[-top_n:]:
        res[features_names[idx]] = importances[idx]

    return res


def get_subset_data(
    filepath: str,
    subset: str,
    patient_id_col: str = 'Patient ID',
    target_col: str = 'ER'
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Get specific subset of data

    Parameters
    ----------
    filepath : str
        Image features path after prepare stage
    patient_id_col : str, optional
        Patient ID col name, by default 'Patient ID'
    target_col : str, optional
        Target col name, by default 'ER'

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str], List[str]]
        Features, targets, features names, patients IDs
    """

    df = pd.read_parquet(filepath)
    df_subset = df[df['subset'] == subset]
    targets = df_subset[target_col].values
    patients_ids = df_subset[patient_id_col].tolist()
    df_subset = df_subset.drop(columns=[patient_id_col, target_col, 'subset'])
    features = df_subset.values
    logger.info(f'Data shape: x_train={features.shape}, y_train={targets.shape}')

    return features, targets, list(df_subset.columns), patients_ids
