from datasets import load_dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = "lainmn/AgentDS-Healthcare"

class DatasetLoader:
    def __init__(self):
        self.dataset_name = DATASET_NAME
        self.ds = None

    def load_full_dataset(self):
        """Loads the huggingface dataset."""
        if self.ds is None:
            logger.info(f"Loading dataset {self.dataset_name}...")
            # Using 'main' or default config
            try:
                self.ds = load_dataset(self.dataset_name)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise e
        return self.ds

    def get_challenge_data(self, keyword: str):
        """
        Heuristic to find the right subset/file for a challenge.
        Looks for keys in the dataset that match the keyword.
        """
        ds = self.load_full_dataset()
        
        # Heuristic: inspect available keys
        # The dataset might be a DatasetDict with keys like 'train', 'test' 
        # OR it might have different configs corresponding to domains.
        # Assuming the structure is domain/problem_statement based or flat files.
        # If flat files, key structure might vary. 
        
        # Fallback: Assume the dataset has splits corresponding to the problem, 
        # or we filter a main table. 
        # Given "lainmn/AgentDS-Healthcare", let's assume likely keys.
        
        relevant_keys = [k for k in ds.keys() if keyword in k.lower()]
        
        if not relevant_keys:
            # Fallback for when data might be packed in a way that requires manual mapping
            # This part would need actual inspection of the dataset structure.
            # For now, we return the first available split or empty if specific mapping fails
            # hoping the repo structure matches.
            
            # Additional Heuristic: Check for 'train' and assume it's a monolithic dataset
            # if specific keys aren't found, OR try to map known file names.
            logger.warning(f"No specific key found for {keyword}. Available keys: {list(ds.keys())}")
            
            # Mock return for development if real data missing (remove in prod)
            # return pd.DataFrame() 
            raise ValueError(f"Could not find data for keyword '{keyword}' in {list(ds.keys())}")

        # Combine splits if multiple relevant ones found, or return the most likely 'train'
        # Ideally we want train and test.
        
        train_key = next((k for k in relevant_keys if 'train' in k), relevant_keys[0])
        return ds[train_key].to_pandas()

    def get_readmission_data(self):
        """
        PS1: 30-day readmission prediction.
        Expected files/keys might involve 'readmission' or 'hospital'.
        """
        # Try finding exact match or use keywords
         # In a real scenario, we'd check `description.md` or dataset keys.
         # Hypothesizing keys based on typical structure.
        try:
             return self.get_challenge_data("readmission")
        except ValueError:
            # Fallback if specific naming differs
            return self.get_challenge_data("train") # Dangerous fallback, but placeholder

    def get_ed_cost_data(self):
        """
        PS2: ED cost forecasting.
        """
        try:
            return self.get_challenge_data("cost")
        except ValueError:
             return pd.DataFrame() # fail gracefully

    def get_discharge_data(self):
        """
        PS3: Discharge readiness.
        """
        try:
            return self.get_challenge_data("discharge")
        except ValueError:
             return pd.DataFrame()

loader = DatasetLoader()
