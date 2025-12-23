import pandas as pd
from backend.commerce.product_recommendation.model.train import train_recommendation_model
from backend.commerce.product_recommendation.model.predict import predict_recommendation
from backend.commerce.product_recommendation.schemas import BatchRecInput

class RecommendationService:
    def train(self):
        try:
            metrics = train_recommendation_model()
            return {"success": True, "metrics": metrics}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, payload: BatchRecInput):
        data = [item.dict() for item in payload.inputs]
        df = pd.DataFrame(data)
        
        preds = predict_recommendation(df)
        
        results = []
        for i, score in enumerate(preds):
            row = data[i]
            results.append({
                "customer_id": row['customer_id'],
                "product_id": row['product_id'],
                "predicted_score": float(score)
            })
        return results

service = RecommendationService()
