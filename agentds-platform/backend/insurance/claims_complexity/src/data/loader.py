import pandas as pd
import os
from src.utils.config import get_full_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_dir = config['paths']['raw_data']
        
    def load_csv(self, filename):
        """
        Load a CSV file from the raw data directory.
        """
        path = get_full_path(os.path.join(self.raw_dir, filename))
        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
            
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from {filename}")
        logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df

    def load_all_data(self):
        """
        Load all claims and policies data (train and test if available).
        """
        data = {}
        files = {
            'train_claims': self.config['data']['train_claims'],
            'train_policies': self.config['data']['train_policies'],
            'test_claims': self.config['data']['test_claims'],
            'test_policies': self.config['data']['test_policies']
        }
        
        for key, filename in files.items():
            try:
                data[key] = self.load_csv(filename)
            except FileNotFoundError:
                logger.warning(f"Optional file {filename} ({key}) not found. Skipping.")
                data[key] = None
        
        return data

if __name__ == "__main__":
    from src.utils.config import load_config
    config = load_config()
    loader = DataLoader(config)
    data = loader.load_all_data()
