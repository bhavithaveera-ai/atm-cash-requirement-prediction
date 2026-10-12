from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / 'datasets'
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / 'models'
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
CLEAN_DATA_PATH = PROCESSED_DIR / 'atm_cash_dataset.csv'
MODEL_PATH = MODELS_DIR / 'atm_cash_model.joblib'
HISTORY_PATH = PROCESSED_DIR / 'prediction_history.json'
SYNTHETIC_DAILY_DATA_PATH = DATASET_DIR / 'synthetic_atm_daily_data.csv'
SYNTHETIC_DAILY_MODEL_PATH = MODELS_DIR / 'synthetic_daily_atm_model.joblib'
SYNTHETIC_DAILY_SUMMARY_PATH = MODELS_DIR / 'synthetic_daily_model_summary.json'
