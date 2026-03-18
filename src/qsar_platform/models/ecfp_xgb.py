import xgboost as xgb
from qsar_platform.models.base import BaseModel

class ECFPXGBModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
        }
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model.load_model(path)
        return model
