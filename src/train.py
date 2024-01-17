"""
Train an XGBClassifier using Optuna
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml
from loguru import logger
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.integration import OptunaSearchCV
from xgboost import XGBClassifier

from src.constants import DICT_INDENT
from src.utils import get_subset_data, get_top_n_feature_importances


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    features_names: List[str],
    estimator: XGBClassifier,
    hparams_grid: Dict[str, Any],
    cv: int = 5,
    n_jobs: int = 5,
    n_trials: int = 100,
    seed: int = 42,
    top_n: int = 10
) -> XGBClassifier:
    param_distributions = {
        'max_depth': IntDistribution(
            low=hparams_grid['max_depth'][0],
            high=hparams_grid['max_depth'][1]
        ),
        'learning_rate': FloatDistribution(
            low=hparams_grid['learning_rate'][0],
            high=hparams_grid['learning_rate'][1]
        ),
        'n_estimators': IntDistribution(
            low=hparams_grid['n_estimators'][0],
            high=hparams_grid['n_estimators'][1]
        )
    }
    search = OptunaSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        cv=cv,
        n_jobs=n_jobs,
        n_trials=n_trials,
        random_state=seed,
        scoring='roc_auc'
    )
    search.fit(x_train, y_train)
    logger.info(f'Best params: {search.best_params_}')
    logger.info(f'Best score: {search.best_score_}')
    best_model = search.best_estimator_
    importances = get_top_n_feature_importances(
        importances=best_model.feature_importances_,
        features_names=features_names,
        top_n=top_n
    )
    logger.info(f'Top-{top_n} features:')
    logger.info(json.dumps(
        {k: str(v) for k, v in importances.items()},
        indent=DICT_INDENT
    ))

    return best_model


if __name__ == '__main__':
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get train data
    x_train, y_train, features_names = get_subset_data(
        filepath=config['prepare']['out_path'],
        subset='train',
        patient_id_col=config['prepare']['patient_id_col'],
        target_col=config['prepare']['target_col']
    )

    # Train
    best_model = train_model(
        x_train=x_train,
        y_train=y_train,
        features_names=features_names,
        estimator=XGBClassifier(
            objective='binary:logistic',
            seed=config['train']['seed']
        ),
        hparams_grid=config['train']['hparams_grid'],
        cv=config['train']['cv'],
        n_jobs=config['train']['n_jobs'],
        n_trials=config['train']['n_trials'],
        seed=config['train']['seed']
    )

    # Save model
    Path(config['train']['out_path']).parent.absolute().mkdir(parents=True, exist_ok=True)
    best_model.save_model(config['train']['out_path'])
