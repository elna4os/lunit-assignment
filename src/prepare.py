"""
Prepare data for train/test:
- Split to train/test
- Append target
- Save everything to single DataFrame
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from loguru import logger


def split_to_train_test(
    df_image_features: pd.DataFrame,
    df_lunit_preds: pd.DataFrame,
    patient_id_col: str = 'Patient ID'
) -> None:
    """
    Split image features to train/test by adding a 'subset' col (inplace)

    Parameters
    ----------
    df_image_features : pd.DataFrame
        Image features
    df_lunit_preds : pd.DataFrame
        Lunit preds
    patient_id_col : str, optional
        Patient ID col name, by default 'Patient ID'
    """

    df_image_features['subset'] = df_image_features[patient_id_col]. \
        isin(df_lunit_preds[patient_id_col]). \
        replace({True: 'test', False: 'train'})
    logger.info(f"Train/test size: {df_image_features['subset'].value_counts().to_dict()}")


def append_target(
    df_image_features: pd.DataFrame,
    df_clinical_features: pd.DataFrame,
    patient_id_col: str = 'Patient ID',
    target_col: str = 'ER'
) -> pd.DataFrame:
    """
    Append target to image features

    Parameters
    ----------
    df_image_features : pd.DataFrame
        Image features
    df_clinical_features : pd.DataFrame
        Clinical and other features
    patient_id_col : str, optional
        Patient ID col name, by default 'Patient ID'
    target_col : str, optional
        Target col name, by default 'ER'

    Returns
    -------
    pd.DataFrame
        Image features with target
    """

    res = df_image_features.merge(
        right=df_clinical_features[[patient_id_col, target_col]],
        on=patient_id_col,
        how='inner'
    )
    logger.info(f"ER on train: {res[res['subset'] == 'train']['ER'].value_counts().to_dict()}")
    logger.info(f"ER on test: {res[res['subset'] == 'test']['ER'].value_counts().to_dict()}")

    return res


def read_excel_data(
    image_features_path: str,
    clinical_features_path: str,
    lunit_preds_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read original data stored in Excel format

    Parameters
    ----------
    image_features_path : str
        Image features file
    clinical_features_path : str
        Clinical and other features file
    lunit_preds_path : str
        Lunit preds file

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Image features DF, clinical and other features DF, Lunit preds DF
    """

    return (
        pd.read_excel(image_features_path),
        pd.read_excel(clinical_features_path, header=[1], skiprows=[2]),
        pd.read_excel(lunit_preds_path)
    )


if __name__ == '__main__':
    with open('params.yaml', 'rb') as f:
        config = yaml.safe_load(f)

    # Read data
    (
        df_image_features,
        df_clinical_features,
        df_lunit_preds
    ) = read_excel_data(
        image_features_path=config['prepare']['image_features_path'],
        clinical_features_path=config['prepare']['clinical_features_path'],
        lunit_preds_path=config['prepare']['lunit_preds_path']
    )

    # Split to train/test
    split_to_train_test(
        df_image_features=df_image_features,
        df_lunit_preds=df_lunit_preds,
        patient_id_col=config['prepare']['patient_id_col']
    )

    df_image_features_out = append_target(
        df_image_features=df_image_features,
        df_clinical_features=df_clinical_features,
        patient_id_col=config['prepare']['patient_id_col'],
        target_col=config['prepare']['target_col']
    )

    # Save
    out_path = Path(config['prepare']['out_path'])
    out_path.parent.absolute().mkdir(parents=True, exist_ok=True)
    df_image_features_out.to_parquet(out_path, index=False)
