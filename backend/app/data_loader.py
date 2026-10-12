import re
from pathlib import Path

import pandas as pd

from backend.app.config import CLEAN_DATA_PATH, DATASET_DIR, PROCESSED_DIR


def get_dataset_files() -> list[Path]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {DATASET_DIR}")

    files = [
        path for path in DATASET_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {'.xls', '.xlsx'}
    ]
    if not files:
        raise FileNotFoundError(f"No Excel files were found in dataset directory: {DATASET_DIR}")
    return sorted(files)

RAW_FIELD_MAP = {
    3: 'atm_on_site',
    4: 'atm_off_site',
    5: 'pos_count',
    6: 'micro_atm_count',
    7: 'bharat_qr_codes',
    8: 'upi_qr_codes',
    9: 'credit_cards',
    10: 'debit_cards',
    11: 'card_payments_pos_volume',
    12: 'card_payments_pos_value',
    13: 'card_payments_online_volume',
    14: 'card_payments_online_value',
    15: 'card_payments_others_volume',
    16: 'card_payments_others_value',
    17: 'cash_withdrawal_at_atm_volume',
    18: 'cash_withdrawal_at_atm_value',
    19: 'cash_withdrawal_at_pos_volume',
    20: 'cash_withdrawal_at_pos_value',
    21: 'cash_withdrawal_online_volume',
    22: 'cash_withdrawal_online_value',
    23: 'cash_withdrawal_others_volume',
    24: 'cash_withdrawal_others_value',
    25: 'cash_withdrawal_total_atm_volume',
    26: 'cash_withdrawal_total_atm_value',
    27: 'cash_withdrawal_total_pos_volume',
    28: 'cash_withdrawal_total_pos_value',
}

TARGET_COLUMN = 'cash_withdrawal_total_atm_value'


def parse_month_from_filename(filename: str) -> str:
    normalized = filename.upper()
    month_names = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
    ]

    for month_name in month_names:
        if month_name.upper() in normalized:
            year_match = re.search(r'(20\d{2})', filename)
            if year_match:
                return f"{month_name} {year_match.group(1)}"

    month_number_match = re.search(r'ATM(?:0?([1-9]|1[0-2]))(20\d{2})', normalized)
    if month_number_match:
        month_index = int(month_number_match.group(1))
        return f"{month_names[month_index - 1]} {month_number_match.group(2)}"

    return filename


def load_raw_dataset() -> pd.DataFrame:
    rows = []
    dataset_files = get_dataset_files()

    for file_path in dataset_files:
        month_label = parse_month_from_filename(file_path.name)
        try:
            df = pd.read_excel(file_path, sheet_name=0, header=None)
        except Exception as exc:  # pragma: no cover - surfaced by caller diagnostics
            raise ValueError(f"Could not read Excel file '{file_path.name}': {exc}") from exc

        for idx in range(8, len(df)):
            row = df.iloc[idx]
            sr_no = row.iloc[1]
            bank_name = row.iloc[2]

            if pd.notna(sr_no) and pd.notna(bank_name) and str(sr_no).strip().isdigit():
                record = {
                    'month': month_label,
                    'date': pd.to_datetime(month_label, format='%B %Y'),
                    'sr_no': int(sr_no),
                    'bank_name': str(bank_name).strip(),
                }
                for idx_col, column_name in RAW_FIELD_MAP.items():
                    record[column_name] = pd.to_numeric(row.iloc[idx_col], errors='coerce')
                rows.append(record)

    if not rows:
        raise ValueError('No usable bank-level rows were found in the raw dataset files.')

    if not CLEAN_DATA_PATH.parent.exists():
        CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    cleaned = pd.DataFrame(rows)
    cleaned['bank_name'] = cleaned['bank_name'].astype(str).str.strip()
    cleaned['cash_requirement_value'] = pd.to_numeric(cleaned[TARGET_COLUMN], errors='coerce').astype(float)
    cleaned = cleaned[cleaned['cash_requirement_value'].notna() & (cleaned['cash_requirement_value'] > 0)].copy()
    cleaned['cash_requirement_inr'] = cleaned['cash_requirement_value'] * 1000
    cleaned['month_number'] = cleaned['date'].dt.month
    cleaned['year'] = cleaned['date'].dt.year
    cleaned['month_label'] = cleaned['date'].dt.strftime('%b %Y')
    cleaned = cleaned.sort_values(['date', 'bank_name']).reset_index(drop=True)

    return cleaned


def save_clean_dataset(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    export_path = CLEAN_DATA_PATH
    df.to_csv(export_path, index=False)
    return export_path


def load_clean_dataset() -> pd.DataFrame:
    clean_path = CLEAN_DATA_PATH
    if clean_path.exists():
        df = pd.read_csv(clean_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'cash_requirement_value' in df.columns:
            filtered = df[df['cash_requirement_value'].notna() & (df['cash_requirement_value'] > 0)].copy()
            if len(filtered) != len(df):
                df = filtered.reset_index(drop=True)
                save_clean_dataset(df)
        return df

    df = load_raw_dataset()
    save_clean_dataset(df)
    return df
