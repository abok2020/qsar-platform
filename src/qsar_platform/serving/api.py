from fastapi import FastAPI, HTTPException
from qsar_platform.contracts.schemas import PredictRequest, PredictResponse
from qsar_platform.serving.inference import InferencePipeline

app = FastAPI(title="QSAR Ensemble API", version="0.1.0")
pipeline = InferencePipeline.load_from_registry(alias="champion")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return pipeline.predict_one(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/predict-batch")
def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
    return [pipeline.predict_one(req) for req in requests]
