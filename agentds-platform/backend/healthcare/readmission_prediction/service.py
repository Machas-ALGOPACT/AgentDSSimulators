from backend.healthcare.readmission_prediction.model.train import train_model
from backend.healthcare.readmission_prediction.model.predict import make_prediction

class ReadmissionService:
    @staticmethod
    def train():
        # Wrapper for training logic
        # In the future, this could be async or Celery task
        return train_model()

    @staticmethod
    def predict(data: list):
        # Wrapper for inference
        # allows adding monitoring, caching, etc.
        return make_prediction(data)

service = ReadmissionService()
