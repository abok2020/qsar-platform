import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from qsar_platform.features.ecfp import smiles_to_ecfp
from qsar_platform.models.ecfp_xgb import ECFPXGBModel
import numpy as np


def train_ecfp_model(df):
    smiles = df["smiles_raw"].tolist()
    y = df["target_value"].values

    X, _ = smiles_to_ecfp(smiles)

    unique_classes, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 samples in each class for stratified CV, got class counts: {dict(zip(unique_classes, counts))}"
        )

    class_counts = {int(k): int(v) for k, v in zip(unique_classes, counts)}
    print(f"Using {n_splits} CV folds based on class counts: {class_counts}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(df))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold}")
        model = ECFPXGBModel()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict_proba(X[valid_idx])
        oof[valid_idx] = preds

    auc = roc_auc_score(y, oof)
    return oof, auc
