from pathlib import Path
import typer
import pandas as pd
import yaml

app = typer.Typer(help="QSAR platform CLI")

def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

@app.command()
def ingest(input: str, output: str) -> None:
    """Ingest a CSV file and write Parquet."""
    _ensure_parent(output)
    df = pd.read_csv(input)
    df.to_parquet(output, index=False)
    typer.echo(f"Wrote {len(df)} rows to {output}")

@app.command()
def standardize(config: str) -> None:
    """Placeholder chemistry standardization stage."""
    cfg = yaml.safe_load(Path(config).read_text())
    typer.echo(f"Loaded standardization config: {cfg.get('name', 'standardization')}")

@app.command()
def split(config: str) -> None:
    """Placeholder split generation stage."""
    cfg = yaml.safe_load(Path(config).read_text())
    typer.echo(f"Loaded split config: {cfg.get('name', 'split')}")

@app.command("train-desc")
def train_desc(input: str, output: str = "data/oof/desc_oof.parquet"):
    import pandas as pd
    from pathlib import Path
    from qsar_platform.training.train_descriptors import train_descriptor_model
    from qsar_platform.utils.mlflow_utils import (
        start_run,
        log_common_params,
        log_metric,
        log_artifact,
    )

    df = pd.read_parquet(input)

    with start_run("train-desc"):
        oof, auc, n_splits = train_descriptor_model(df)

        df["desc_lgbm_oof"] = oof

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)

        log_common_params("desc_lgbm", n_splits, input, output)
        log_metric("roc_auc", auc)
        log_artifact(output)

        print(f"[DESC] AUC: {auc:.4f}")
        print(f"Saved to {output}")

@app.command("train-model")
def train_model(config: str) -> None:
    """Placeholder single-model training stage."""
    cfg = yaml.safe_load(Path(config).read_text())
    typer.echo(f"Training model: {cfg.get('name', 'unknown_model')}")

@app.command("train-ecfp")
def train_ecfp(input: str, output: str = "data/oof/ecfp_oof.parquet") -> None:
    import pandas as pd
    from pathlib import Path
    from qsar_platform.training.train_ecfp import train_ecfp_model

    df = pd.read_parquet(input)
    oof, auc, n_splits = train_ecfp_model(df)

    out_df = df.copy()
    out_df["ecfp_xgb_oof"] = oof

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)

    log_common_params("ecfp_xgb", n_splits, input, output)
    log_metric("roc_auc", auc)
    log_artifact(output)

    print(f"Training complete")
    print(f"OOF ROC-AUC: {auc:.4f}")
    print(f"Saved OOF predictions to {output}")

@app.command("train-all")
def train_all(params: str) -> None:
    """Placeholder multi-model training stage."""
    cfg = yaml.safe_load(Path(params).read_text())
    models = cfg.get("active", {}).get("base_models", [])
    typer.echo(f"Training {len(models)} base models")

@app.command("fit-ensemble")
def fit_ensemble(config: str) -> None:
    """Placeholder ensemble fitting stage."""
    cfg = yaml.safe_load(Path(config).read_text())
    typer.echo(f"Fitting ensemble: {cfg.get('name', 'ensemble')}")

@app.command("ensemble-oof")
def ensemble_oof(
    ecfp_input: str = "data/oof/ecfp_oof.parquet",
    desc_input: str = "data/oof/desc_oof.parquet",
    output: str = "data/oof/ensemble_oof.parquet",
) -> None:
    import mlflow
    from sklearn.metrics import roc_auc_score
    from qsar_platform.ensemble.weighted_avg import fit_weighted_average
    from qsar_platform.utils.mlflow_utils import start_run, log_metric, log_artifact

    with start_run("ensemble-oof"):
        merged = fit_weighted_average(
            ecfp_path=ecfp_input,
            desc_path=desc_input,
            output_path=output,
        )

        auc = roc_auc_score(merged["target_value"], merged["ensemble_oof"])
        mlflow.log_param("model_name", "weighted_avg")
        mlflow.log_param("base_models", "ecfp_xgb,desc_lgbm")
        log_metric("roc_auc", auc)
        log_artifact(output)

@app.command()
def evaluate(run_group: str) -> None:
    typer.echo(f"Evaluating run group: {run_group}")

@app.command()
def register(run_id: str) -> None:
    typer.echo(f"Registering run: {run_id}")

if __name__ == "__main__":
    app()

