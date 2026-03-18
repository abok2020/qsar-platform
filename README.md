# qsar-platform

A production-oriented SAR/QSAR ensemble platform scaffold.

## Included
- Config-driven project layout
- CLI skeleton
- Base model interface
- FastAPI inference service
- Pydantic request/response contracts
- GitHub Actions workflow skeletons
- Dockerfile
- pyproject.toml

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,train,chem]
uvicorn qsar_platform.serving.api:app --reload
```

## Example commands

```bash
python -m qsar_platform.cli ingest --input data/raw/assay.csv --output data/interim/ingested.parquet
python -m qsar_platform.cli standardize --config configs/data/standardization.yaml
python -m qsar_platform.cli split --config configs/splits/scaffold.yaml
python -m qsar_platform.cli train-all --params params.yaml
python -m qsar_platform.cli fit-ensemble --config configs/ensemble/weighted_avg.yaml
```
