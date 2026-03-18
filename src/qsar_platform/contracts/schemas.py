from typing import Literal
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    compound_id: str = Field(..., description="Unique compound identifier")
    smiles: str = Field(..., description="Input SMILES string")
    assay_id: str = Field(..., description="Endpoint or assay identifier")

class PredictResponse(BaseModel):
    compound_id: str
    assay_id: str
    model_version: str
    prediction: float
    prediction_type: Literal["probability", "score", "value"]
    base_models: dict[str, float]
    ensemble_std: float
    applicability_domain: Literal["inside", "borderline", "outside"]
