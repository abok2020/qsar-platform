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

@app.command("train-model")
def train_model(config: str) -> None:
    """Placeholder single-model training stage."""
    cfg = yaml.safe_load(Path(config).read_text())
    typer.echo(f"Training model: {cfg.get('name', 'unknown_model')}")

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

@app.command()
def evaluate(run_group: str) -> None:
    typer.echo(f"Evaluating run group: {run_group}")

@app.command()
def register(run_id: str) -> None:
    typer.echo(f"Registering run: {run_id}")

if __name__ == "__main__":
    app()
