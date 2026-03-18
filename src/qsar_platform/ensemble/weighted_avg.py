import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


def fit_weighted_average(
    ecfp_path: str = "data/oof/ecfp_oof.parquet",
    desc_path: str = "data/oof/desc_oof.parquet",
    output_path: str = "data/oof/ensemble_oof.parquet",
) -> pd.DataFrame:
    ecfp = pd.read_parquet(ecfp_path)
    desc = pd.read_parquet(desc_path)

    merged = ecfp.merge(
        desc[["compound_id", "desc_lgbm_oof"]],
        on="compound_id",
        how="inner",
    )

    merged["ensemble_oof"] = (
        0.5 * merged["ecfp_xgb_oof"] + 0.5 * merged["desc_lgbm_oof"]
    )

    auc = roc_auc_score(merged["target_value"], merged["ensemble_oof"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    print(f"[ENSEMBLE] AUC: {auc:.4f}")
    print(f"Saved ensemble predictions to {output_path}")

    return merged
