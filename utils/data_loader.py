import pandas as pd
from utils.logger import setup_logger

logger = setup_logger()
def load_data(filepath: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"❌ File not found at: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"❌ The file at {filepath} is empty or corrupted.")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while loading data: {e}")
        raise
