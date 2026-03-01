from datasets import load_dataset
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = "lainmn/AgentDS-Healthcare"

import os
from pathlib import Path

class DatasetLoader:
    def __init__(self):
        self.base_url = "hf://datasets/lainmn/AgentDS-Healthcare/Healthcare"
        # Points to d:\2025\AgentDS\AgentDSSimulators\agentds-platform\backend\data\healthcare
        self.local_dir = Path(__file__).parent.parent.parent / "data" / "healthcare"
        
        # Mapping based on user's manual files
        self.challenges = {
            "readmission": {
                "train": "admissions_train.csv",
                "test": "admissions_test.csv" 
            },
            "ed_cost": {
                "train": "ed_cost_train.csv",
                "test": "ed_cost_test.csv"
            },
            "discharge_readiness": {
                "train": "stays_train.csv",
                "test": "stays_test.csv"
            }
        }

    def load_challenge_data(self, challenge_key: str, split: str = "train") -> pd.DataFrame:
        """
        Loads data from local directory if exists, else defaults to HF.
        """
        if challenge_key not in self.challenges:
            raise ValueError(f"Unknown challenge key: {challenge_key}")
            
        file_name = self.challenges[challenge_key].get(split)
        if not file_name:
             file_name = f"{challenge_key}_{split}.csv"

        # 1. Check Local
        local_path = self.local_dir / file_name
        if local_path.exists():
            logger.info(f"Loading local file: {local_path}")
            return pd.read_csv(local_path)
            
        # 2. Fallback to HF
        full_url = f"{self.base_url}/{file_name}"
        logger.info(f"Local file not found ({local_path}). Downloading from {full_url}...")
        
        try:
            ds = load_dataset("csv", data_files={"data": full_url}, split="data")
            return ds.to_pandas()
        except Exception as e:
            logger.error(f"Failed to load {file_name}: {e}")
            raise e

    def get_readmission_data(self):
        # NOTE: If patients.csv is needed for proper features, merge logic would go here.
        # For now, returning the main admission table.
        df = self.load_challenge_data("readmission", "train")
        
        # Optional: Try to join patients if exists locally
        patients_path = self.local_dir / "patients.csv"
        if patients_path.exists() and "subject_id" in df.columns:
            try:
                patients = pd.read_csv(patients_path)
                df = df.merge(patients, on="subject_id", how="left")
                logger.info("Merged with local patients.csv")
            except Exception as e:
                logger.warning(f"Failed to merge patients.csv: {e}")
                
        return df

    def get_ed_cost_data(self):
        return self.load_challenge_data("ed_cost", "train")

    def get_discharge_data(self):
        return self.load_challenge_data("discharge_readiness", "train")

loader = DatasetLoader()
