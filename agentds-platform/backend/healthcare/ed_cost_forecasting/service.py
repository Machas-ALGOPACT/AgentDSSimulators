from backend.healthcare.ed_cost_forecasting.model.train import train_model
from backend.healthcare.ed_cost_forecasting.model.predict import make_prediction

class EDCostService:
    @staticmethod
    def train():
        return train_model()

    @staticmethod
    def predict(data: list):
        return make_prediction(data)

service = EDCostService()
