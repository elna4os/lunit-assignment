"""
Load test data, make predictions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from src.utils import get_subset_data


def predict(
    model: XGBClassifier,
    x_test: np.ndarray
) -> np.ndarray:
    """
    Predict scores on test data

    Parameters
    ----------
    model : XGBClassifier
        Estimator
    x_test : np.ndarray
        Test features

    Returns
    -------
    np.ndarray
        Test scores
    """

    return model.predict_proba(x_test)


if __name__ == '__main__':
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get test data
    x_test, y_test, features_names, test_ids = get_subset_data(
        filepath=config['prepare']['out_path'],
        subset='test',
        patient_id_col=config['prepare']['patient_id_col'],
        target_col=config['prepare']['target_col']
    )

    # Load model
    model = XGBClassifier()
    model.load_model(config['train']['out_path'])

    # Predict
    test_preds = predict(
        model=model,
        x_test=x_test
    )
    df_preds = pd.DataFrame({
        config['prepare']['patient_id_col']: test_ids,
        'y_score_0': test_preds[:, 0],
        'y_score_1': test_preds[:, 1],
        'y_true': y_test
    })

    # Save
    Path(config['test']['out_path']).parent.absolute().mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(config['test']['out_path'], index=False)
