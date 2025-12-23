import logging
import pandas as pd
from datasets import load_dataset
from backend.commerce.shared.constants import (
    DATASET_FILES, 
    TASK_DEMAND_FORECASTING, 
    TASK_PRODUCT_RECOMMENDATION, 
    TASK_COUPON_REDEMPTION
)

logger = logging.getLogger(__name__)

HF_DATASET_ID = "lainmn/AgentDS-Commerce"

def load_commerce_dataset(task_name: str, split: str = "train") -> pd.DataFrame:
    """
    Loads the specific CSV file associated with a commerce task from Hugging Face.
    
    Args:
        task_name: One of the constants defined (e.g., TASK_DEMAND_FORECASTING).
        split: 'train' or 'test'. Defaults to 'train'.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    if task_name not in DATASET_FILES:
        raise ValueError(f"Unknown task name: {task_name}")

    filename = DATASET_FILES[task_name]
    
    # Adjust filename for test split if necessary
    # The prompt implies standard split names or files. 
    # Based on exploration, files are like 'sales_history_train.csv'.
    # We will assume 'train' implies the '_train.csv' version unless we identify '_test.csv'.
    # For now, let's map strictly based on the exploration findings.
    
    target_file = filename
    if split == "test":
        target_file = filename.replace("_train.csv", "_test.csv")

    logger.info(f"Loading dataset for task={task_name}, split={split}, file={target_file}")

    try:
        # Load specific file using data_files
        # We use split="train" because even if we load a single file, HF datasets often defaults to a 'train' split key 
        # unless we specify otherwise. When data_files is a dict {'train': path}, ds['train'] is valid.
        
        data_files = {"train": target_file}
        
        # We trust remote code false by default, but for CSV it's fine.
        ds = load_dataset(HF_DATASET_ID, data_files=data_files, split="train")
        
        df = ds.to_pandas()
        logger.info(f"Successfully loaded {len(df)} rows for {task_name}")
        return df

    except Exception as e:
        logger.error(f"Failed to load dataset for {task_name}: {e}")
        # Fallback or re-raise. For an enterprise app, we re-raise.
        raise RuntimeError(f"Could not load dataset {HF_DATASET_ID}/{target_file}: {e}")
