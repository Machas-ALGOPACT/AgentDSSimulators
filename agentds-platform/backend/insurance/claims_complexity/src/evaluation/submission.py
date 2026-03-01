import pandas as pd
import os
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_submission(model, X_test, test_ids, le_classes, output_dir='outputs'):
    """
    Generate submission file.
    """
    logger.info("Generating submission file...")
    
    y_pred = model.predict(X_test)
    y_pred_labels = [le_classes[i] for i in y_pred]
    
    submission = pd.DataFrame({
        'ClaimID': test_ids,
        'ClaimComplexityLabel': y_pred_labels
    })
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_{timestamp}.csv"
    path = os.path.join(output_dir, filename)
    
    submission.to_csv(path, index=False)
    logger.info(f"Submission saved to {path} with {len(submission)} rows.")
    return path
