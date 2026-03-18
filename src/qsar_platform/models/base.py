from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    name: str

    @abstractmethod
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        raise NotImplementedError
