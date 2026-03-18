import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from qsar_platform.features.descriptors import smiles_to_descriptors
from qsar_platform.models.desc_lgbm import DescLGBMModel


def train_descriptor_model(df):
    smiles = df["smiles_raw"].tolist()
    y = df["target_value"].values

    X = smiles_to_descriptors(smiles)

    unique, counts = np.unique(y, return_counts=True)
    n_splits = min(5, counts.min())

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(df))

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print(f"[DESC] Fold {fold}")

        model = DescLGBMModel()
        model.fit(X[train_idx], y[train_idx])

        preds = model.predict_proba(X[valid_idx])
        oof[valid_idx] = preds

    auc = roc_auc_score(y, oof)
    return oof, auc, n_splits
