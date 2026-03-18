import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


def fit_weighted_average(
    ecfp_path: str = "data/oof/ecfp_oof.parquet",
    desc_path: str = "data/oof/desc_oof.parquet",
    output_path: str = "data/oof/ensemble_oof.parquet",
    weights_path: str = "models/ensemble/weights.json",
) -> pd.DataFrame:
    ecfp = pd.read_parquet(ecfp_path)
    desc = pd.read_parquet(desc_path)

    merged = ecfp.merge(
        desc[["compound_id", "desc_lgbm_oof"]],
        on="compound_id",
        how="inner",
    )

    ecfp_auc = roc_auc_score(merged["target_value"], merged["ecfp_xgb_oof"])
    desc_auc = roc_auc_score(merged["target_value"], merged["desc_lgbm_oof"])

    total = ecfp_auc + desc_auc
    w_ecfp = ecfp_auc / total
    w_desc = desc_auc / total

    merged["ensemble_oof"] = (
        w_ecfp * merged["ecfp_xgb_oof"] + w_desc * merged["desc_lgbm_oof"]
    )

    auc = roc_auc_score(merged["target_value"], merged["ensemble_oof"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ecfp_xgb": w_ecfp,
                "desc_lgbm": w_desc,
                "ensemble_auc": auc,
            },
            f,
            indent=2,
        )

    print(f"[ENSEMBLE] ECFP AUC: {ecfp_auc:.4f}")
    print(f"[ENSEMBLE] DESC AUC: {desc_auc:.4f}")
    print(f"[ENSEMBLE] Weights: ecfp_xgb={w_ecfp:.4f}, desc_lgbm={w_desc:.4f}")
    print(f"[ENSEMBLE] AUC: {auc:.4f}")
    print(f"Saved ensemble predictions to {output_path}")
    print(f"Saved ensemble weights to {weights_path}")

    return merged
