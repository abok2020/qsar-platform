import lightgbm as lgb
from qsar_platform.models.base import BaseModel

class DescLGBMModel(BaseModel):
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31
        )

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        import joblib
        model = cls()
        model.model = joblib.load(path)
        return model
