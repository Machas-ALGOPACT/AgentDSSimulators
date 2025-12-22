from backend.healthcare.discharge_readiness.model.train import train_model
from backend.healthcare.discharge_readiness.model.predict import make_prediction

class DischargeReadinessService:
    @staticmethod
    def train():
        return train_model()

    @staticmethod
    def predict(data: list):
        return make_prediction(data)

service = DischargeReadinessService()
