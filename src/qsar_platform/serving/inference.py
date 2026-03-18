from qsar_platform.contracts.schemas import PredictRequest, PredictResponse

class InferencePipeline:
    def __init__(self, model_version: str = "dev"):
        self.model_version = model_version

    @classmethod
    def load_from_registry(cls, alias: str = "champion") -> "InferencePipeline":
        return cls(model_version=f"{alias}-local")

    def predict_one(self, req: PredictRequest) -> PredictResponse:
        base_models = {
            "ecfp_xgb": 0.71,
            "desc_lgbm": 0.68,
            "hf_embed_xgb": 0.77,
        }
        prediction = sum(base_models.values()) / len(base_models)
        return PredictResponse(
            compound_id=req.compound_id,
            assay_id=req.assay_id,
            model_version=self.model_version,
            prediction=prediction,
            prediction_type="probability",
            base_models=base_models,
            ensemble_std=0.04,
            applicability_domain="inside",
        )
