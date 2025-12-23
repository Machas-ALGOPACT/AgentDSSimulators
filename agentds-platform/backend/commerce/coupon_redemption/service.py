import pandas as pd
from backend.commerce.coupon_redemption.model.train import train_coupon_model
from backend.commerce.coupon_redemption.model.predict import predict_coupon
from backend.commerce.coupon_redemption.schemas import BatchCouponInput

class CouponService:
    def train(self):
        try:
            metrics = train_coupon_model()
            return {"success": True, "metrics": metrics}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, payload: BatchCouponInput):
        data = [item.dict() for item in payload.inputs]
        df = pd.DataFrame(data)
        
        preds, probs = predict_coupon(df)
        
        results = []
        for i, prob in enumerate(probs):
            row = data[i]
            results.append({
                "customer_id": row['customer_id'],
                "offer_id": row['offer_id'],
                "redemption_probability": float(prob),
                "will_redeem": bool(preds[i])
            })
        return results

service = CouponService()
