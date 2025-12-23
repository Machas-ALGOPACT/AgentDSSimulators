import pandas as pd
from backend.commerce.demand_forecasting.model.train import train_demand_model
from backend.commerce.demand_forecasting.model.predict import predict_demand
from backend.commerce.demand_forecasting.schemas import BatchForecastInput

class ForecastingService:
    def train(self):
        try:
            metrics = train_demand_model()
            return {"success": True, "metrics": metrics}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, payload: BatchForecastInput):
        data = [item.dict() for item in payload.inputs]
        df = pd.DataFrame(data)
        
        predictions = predict_demand(df)
        
        # Zip back with identifiers
        results = []
        for i, pred in enumerate(predictions):
            row = data[i]
            results.append({
                "sku_id": row['sku_id'],
                "week": row['week'],
                "predicted_units_sold": float(pred)
            })
        return results

service = ForecastingService()
