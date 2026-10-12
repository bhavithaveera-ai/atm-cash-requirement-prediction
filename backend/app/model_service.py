import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from backend.app.config import (
    HISTORY_PATH,
    PROCESSED_DIR,
    SYNTHETIC_DAILY_DATA_PATH,
    SYNTHETIC_DAILY_MODEL_PATH,
    SYNTHETIC_DAILY_SUMMARY_PATH,
)
from backend.app.data_loader import load_clean_dataset

MODEL_TARGET = 'cash_requirement_value'
MODEL_FEATURES = [
    'atm_on_site',
    'atm_off_site',
    'pos_count',
    'micro_atm_count',
    'bharat_qr_codes',
    'upi_qr_codes',
    'credit_cards',
    'debit_cards',
    'card_payments_pos_volume',
    'card_payments_pos_value',
    'card_payments_online_volume',
    'card_payments_online_value',
    'card_payments_others_volume',
    'card_payments_others_value',
    'cash_withdrawal_at_atm_volume',
    'cash_withdrawal_at_atm_value',
    'cash_withdrawal_at_pos_volume',
    'cash_withdrawal_at_pos_value',
    'cash_withdrawal_online_volume',
    'cash_withdrawal_online_value',
    'cash_withdrawal_others_volume',
    'cash_withdrawal_others_value',
    'cash_withdrawal_total_atm_volume',
    'cash_withdrawal_total_pos_volume',
    'cash_withdrawal_total_pos_value',
    'month_number',
    'year',
]

DAILY_TARGET = 'Withdrawals'
DAILY_FEATURES = [
    'ATM_ID',
    'Location',
    'DayOfWeek',
    'IsHoliday',
    'WeatherImpact',
    'day_of_month',
    'month',
    'year',
    'lag_1',
    'lag_7',
    'rolling_3',
    'rolling_7',
]


def build_pipeline() -> Pipeline:
    numeric_cols = MODEL_FEATURES
    categorical_cols = ['bank_name']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def train_model_and_metrics():
    df = load_clean_dataset().copy()
    df = df.sort_values(['date', 'bank_name']).reset_index(drop=True)

    feature_columns = MODEL_FEATURES + ['bank_name']
    X = df[feature_columns]
    y = df[MODEL_TARGET]

    cutoff = int(len(df) * 0.8)
    X_train = X.iloc[:cutoff]
    X_test = X.iloc[cutoff:]
    y_train = y.iloc[:cutoff]
    y_test = y.iloc[cutoff:]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    result = {
        'model_name': 'RandomForestRegressor',
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
        },
        'actual': y_test.tolist(),
        'predicted': preds.tolist(),
        'bank_names': X_test['bank_name'].tolist(),
        'date_labels': df.iloc[cutoff:]['date'].dt.strftime('%b %Y').tolist(),
    }

    model_path = Path(__file__).resolve().parents[2] / 'models' / 'atm_cash_model.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return result


def load_model():
    model_path = Path(__file__).resolve().parents[2] / 'models' / 'atm_cash_model.joblib'
    if not model_path.exists():
        train_model_and_metrics()
    return joblib.load(model_path)


def ensure_history_file() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text('[]', encoding='utf-8')


def append_prediction_record(record: dict) -> None:
    ensure_history_file()
    records = json.loads(HISTORY_PATH.read_text(encoding='utf-8') or '[]')
    records.append(record)
    HISTORY_PATH.write_text(json.dumps(records, indent=2), encoding='utf-8')


def predict_cash_requirement(payload: dict) -> dict:
    model = load_model()
    record = {
        'bank_name': payload['bank_name'],
        'atm_on_site': float(payload.get('atm_on_site', 0)),
        'atm_off_site': float(payload.get('atm_off_site', 0)),
        'pos_count': float(payload.get('pos_count', 0)),
        'micro_atm_count': float(payload.get('micro_atm_count', 0)),
        'bharat_qr_codes': float(payload.get('bharat_qr_codes', 0)),
        'upi_qr_codes': float(payload.get('upi_qr_codes', 0)),
        'credit_cards': float(payload.get('credit_cards', 0)),
        'debit_cards': float(payload.get('debit_cards', 0)),
        'card_payments_pos_volume': float(payload.get('card_payments_pos_volume', 0)),
        'card_payments_pos_value': float(payload.get('card_payments_pos_value', 0)),
        'card_payments_online_volume': float(payload.get('card_payments_online_volume', 0)),
        'card_payments_online_value': float(payload.get('card_payments_online_value', 0)),
        'card_payments_others_volume': float(payload.get('card_payments_others_volume', 0)),
        'card_payments_others_value': float(payload.get('card_payments_others_value', 0)),
        'cash_withdrawal_at_atm_volume': float(payload.get('cash_withdrawal_at_atm_volume', 0)),
        'cash_withdrawal_at_atm_value': float(payload.get('cash_withdrawal_at_atm_value', 0)),
        'cash_withdrawal_at_pos_volume': float(payload.get('cash_withdrawal_at_pos_volume', 0)),
        'cash_withdrawal_at_pos_value': float(payload.get('cash_withdrawal_at_pos_value', 0)),
        'cash_withdrawal_online_volume': float(payload.get('cash_withdrawal_online_volume', 0)),
        'cash_withdrawal_online_value': float(payload.get('cash_withdrawal_online_value', 0)),
        'cash_withdrawal_others_volume': float(payload.get('cash_withdrawal_others_volume', 0)),
        'cash_withdrawal_others_value': float(payload.get('cash_withdrawal_others_value', 0)),
        'cash_withdrawal_total_atm_volume': float(payload.get('cash_withdrawal_total_atm_volume', 0)),
        'cash_withdrawal_total_pos_volume': float(payload.get('cash_withdrawal_total_pos_volume', 0)),
        'cash_withdrawal_total_pos_value': float(payload.get('cash_withdrawal_total_pos_value', 0)),
        'year': int(payload.get('year', pd.Timestamp.now().year)),
    }

    date_obj = pd.to_datetime(payload.get('month', pd.Timestamp.now().strftime('%b %Y')))
    record['month_number'] = int(date_obj.month)
    input_df = pd.DataFrame([record])
    for col in MODEL_FEATURES:
        if col not in input_df.columns:
            input_df[col] = 0

    prediction_value = float(model.predict(input_df[['bank_name'] + MODEL_FEATURES])[0])
    prediction_inr = prediction_value * 1000
    return {
        'cash_requirement_value': float(prediction_value),
        'cash_requirement_inr': float(prediction_inr),
        'month': date_obj.strftime('%b %Y'),
        'bank_name': payload['bank_name'],
    }


def _normalize_daily_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {}
    for column in df.columns:
        tokens = ''.join(ch.lower() if ch.isalnum() else '' for ch in str(column))
        normalized[column] = tokens

    aliases = {
        'date': 'Date',
        'atm': 'ATM_ID',
        'atmid': 'ATM_ID',
        'atmidentifier': 'ATM_ID',
        'location': 'Location',
        'withdrawals': 'Withdrawals',
        'withdrawalamount': 'Withdrawals',
        'deposits': 'Deposits',
        'dayofweek': 'DayOfWeek',
        'weekday': 'DayOfWeek',
        'isholiday': 'IsHoliday',
        'holiday': 'IsHoliday',
        'weatherimpact': 'WeatherImpact',
        'weather': 'WeatherImpact',
    }

    rename_map = {}
    for original, token in normalized.items():
        candidate = aliases.get(token)
        if candidate is not None:
            rename_map[original] = candidate
    return df.rename(columns=rename_map)


def load_synthetic_daily_dataset() -> pd.DataFrame:
    if not SYNTHETIC_DAILY_DATA_PATH.exists():
        legacy_path = Path(__file__).resolve().parents[2] / 'datasets' / 'synthetic_atm_cash_data.csv'
        if legacy_path.exists():
            SYNTHETIC_DAILY_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, SYNTHETIC_DAILY_DATA_PATH)

    if not SYNTHETIC_DAILY_DATA_PATH.exists():
        raise FileNotFoundError(f'Synthetic daily ATM dataset is missing: {SYNTHETIC_DAILY_DATA_PATH}')

    df = pd.read_csv(SYNTHETIC_DAILY_DATA_PATH)
    df = _normalize_daily_columns(df)

    required_columns = ['Date', 'ATM_ID', 'Location', 'Withdrawals', 'DayOfWeek', 'IsHoliday', 'WeatherImpact']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f'Synthetic daily ATM dataset is missing required columns: {missing_columns}')

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).copy()
    df = df.drop_duplicates().sort_values(['ATM_ID', 'Date']).reset_index(drop=True)

    for column in ['Withdrawals', 'Deposits', 'DayOfWeek', 'IsHoliday', 'WeatherImpact']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    df['ATM_ID'] = df['ATM_ID'].astype(str).str.strip()
    df['Location'] = df['Location'].astype(str).str.strip()
    df['day_of_month'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day_of_week'] = df['Date'].dt.dayofweek

    df['lag_1'] = df.groupby('ATM_ID')['Withdrawals'].transform(lambda series: series.shift(1))
    df['lag_7'] = df.groupby('ATM_ID')['Withdrawals'].transform(lambda series: series.shift(7))
    df['rolling_3'] = df.groupby('ATM_ID')['Withdrawals'].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean(),
    )
    df['rolling_7'] = df.groupby('ATM_ID')['Withdrawals'].transform(
        lambda series: series.shift(1).rolling(window=7, min_periods=1).mean(),
    )

    atm_means = df.groupby('ATM_ID')['Withdrawals'].transform('mean')
    for column in ['lag_1', 'lag_7', 'rolling_3', 'rolling_7']:
        df[column] = df[column].fillna(atm_means)

    return df


def build_daily_pipeline() -> Pipeline:
    categorical_cols = ['ATM_ID', 'Location']
    numeric_cols = ['DayOfWeek', 'IsHoliday', 'WeatherImpact', 'day_of_month', 'month', 'year', 'lag_1', 'lag_7', 'rolling_3', 'rolling_7']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]), categorical_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


def train_synthetic_daily_model_and_metrics() -> dict:
    df = load_synthetic_daily_dataset().copy()
    df = df[(df['Date'] >= pd.Timestamp('2023-01-01')) & (df['Date'] <= pd.Timestamp('2023-12-31'))].copy()

    if df.empty:
        raise ValueError('The synthetic daily ATM dataset does not contain any rows in the required 2023 training window.')

    if df['ATM_ID'].nunique() < 2:
        raise ValueError('The synthetic daily ATM dataset does not contain enough unique ATM IDs for a valid model fit.')

    cutoff = df['Date'].quantile(0.8)
    train_df = df[df['Date'] <= cutoff].copy()
    test_df = df[df['Date'] > cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError('The 2023-only synthetic daily dataset needs more rows to support a chronological train/test split.')

    X_train = train_df[DAILY_FEATURES]
    y_train = train_df[DAILY_TARGET]
    X_test = test_df[DAILY_FEATURES]
    y_test = test_df[DAILY_TARGET]

    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)

    rf_model = build_daily_pipeline()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    baseline_metrics = {
        'mae': float(mean_absolute_error(y_test, baseline_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        'r2_score': float(r2_score(y_test, baseline_pred)),
    }

    rf_metrics = {
        'mae': float(mean_absolute_error(y_test, rf_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        'r2_score': float(r2_score(y_test, rf_pred)),
    }

    selected_model = 'RandomForestRegressor'
    if rf_metrics['r2_score'] < baseline_metrics['r2_score']:
        selected_model = 'DummyRegressor'

    summary = {
        'dataset_path': str(SYNTHETIC_DAILY_DATA_PATH),
        'dataset_rows': int(len(df)),
        'dataset_columns': list(df.columns),
        'date_range': {
            'start': df['Date'].min().strftime('%Y-%m-%d'),
            'end': df['Date'].max().strftime('%Y-%m-%d'),
        },
        'atm_count': int(df['ATM_ID'].nunique()),
        'locations': sorted(df['Location'].unique().tolist()),
        'selected_model': selected_model,
        'model_name': selected_model,
        'baseline_metrics': baseline_metrics,
        'model_metrics': rf_metrics,
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'notes': 'Synthetic daily ATM model was trained on 2023 data only for demonstration purposes; it does not represent a verified RBI or INR-denominated forecast.',
    }

    SYNTHETIC_DAILY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if selected_model == 'DummyRegressor':
        joblib.dump(baseline_model, SYNTHETIC_DAILY_MODEL_PATH)
    else:
        joblib.dump(rf_model, SYNTHETIC_DAILY_MODEL_PATH)

    SYNTHETIC_DAILY_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DAILY_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def load_daily_model():
    if not SYNTHETIC_DAILY_MODEL_PATH.exists():
        train_synthetic_daily_model_and_metrics()
    return joblib.load(SYNTHETIC_DAILY_MODEL_PATH)


def _build_daily_prediction_features(atm_id: str, prediction_date: pd.Timestamp, dataset: pd.DataFrame) -> dict:
    atm_history = dataset[dataset['ATM_ID'] == atm_id].sort_values('Date').copy()
    if atm_history.empty:
        raise ValueError(f'ATM {atm_id} is not supported by the synthetic daily dataset.')

    recent_history = atm_history[atm_history['Date'] < prediction_date].copy()
    if recent_history.empty:
        recent_history = atm_history.head(7).copy()

    if recent_history.empty:
        raise ValueError(f'No usable historical observations exist for ATM {atm_id}.')

    recent_values = recent_history['Withdrawals'].dropna()
    lag_1 = float(recent_values.iloc[-1]) if not recent_values.empty else float(atm_history['Withdrawals'].mean())
    lag_7 = float(recent_values.iloc[-7]) if len(recent_values) >= 7 else float(recent_values.mean()) if not recent_values.empty else float(atm_history['Withdrawals'].mean())
    rolling_3 = float(recent_values.tail(3).mean()) if len(recent_values) >= 1 else float(atm_history['Withdrawals'].mean())
    rolling_7 = float(recent_values.tail(7).mean()) if len(recent_values) >= 1 else float(atm_history['Withdrawals'].mean())

    location = str(atm_history['Location'].mode().iloc[0]).strip() if not atm_history['Location'].mode().empty else 'Unknown'
    weather_impact = int(round(float(atm_history['WeatherImpact'].mean()))) if 'WeatherImpact' in atm_history.columns else 0
    is_holiday = int(prediction_date.dayofweek >= 5)

    feature_row = {
        'ATM_ID': atm_id,
        'Location': location,
        'DayOfWeek': int(prediction_date.dayofweek),
        'IsHoliday': is_holiday,
        'WeatherImpact': weather_impact,
        'day_of_month': int(prediction_date.day),
        'month': int(prediction_date.month),
        'year': int(prediction_date.year),
        'lag_1': float(lag_1),
        'lag_7': float(lag_7),
        'rolling_3': float(rolling_3),
        'rolling_7': float(rolling_7),
    }
    return feature_row


def predict_daily_atm_cash(payload: dict) -> dict:
    dataset = load_synthetic_daily_dataset()
    atm_id = str(payload.get('atm_id', '')).strip()
    prediction_date = pd.to_datetime(payload.get('prediction_date'))

    if pd.isna(prediction_date):
        raise ValueError('prediction_date must be a valid date in YYYY-MM-DD format.')

    if not atm_id:
        raise ValueError('atm_id is required.')

    valid_atms = sorted(dataset['ATM_ID'].dropna().astype(str).unique().tolist())
    if atm_id not in valid_atms:
        raise ValueError(f'ATM {atm_id} is not available in the synthetic daily dataset.')

    feature_row = _build_daily_prediction_features(atm_id, prediction_date, dataset)
    model = load_daily_model()
    prediction_value = float(model.predict(pd.DataFrame([feature_row]))[0])

    return {
        'atm_id': atm_id,
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'predicted_cash_requirement_inr': float(prediction_value),
        'currency_unit': 'synthetic demo units',
        'dataset_kind': 'synthetic_daily_demo',
        'message': 'Synthetic Demo Prediction — Trained on 2023 Data. This synthetic output is not a verified RBI forecast and should not be treated as real INR-denominated cash demand.',
    }


if __name__ == '__main__':
    summary = train_synthetic_daily_model_and_metrics()
    print(json.dumps(summary, indent=2))
