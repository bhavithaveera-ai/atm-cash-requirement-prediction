import json
import logging
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import DATASET_DIR, HISTORY_PATH
from backend.app.data_loader import get_dataset_files, load_clean_dataset
from backend.app.model_service import (
    append_prediction_record,
    load_synthetic_daily_dataset,
    predict_cash_requirement,
    predict_daily_atm_cash,
    train_model_and_metrics,
)
from backend.app.schemas import PredictionPayload

logger = logging.getLogger('atm_cash_api')
logging.basicConfig(level=logging.INFO)

app = FastAPI(title='ATM Cash Requirement Prediction API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


def dataset_summary():
    dataset_files = get_dataset_files()
    df = load_clean_dataset()
    return {
        'dataset_files': len(dataset_files),
        'records_loaded': int(len(df)),
        'unique_banks': int(df['bank_name'].nunique()),
        'months_loaded': int(df['month_label'].nunique()),
        'dataset_loaded': True,
    }


@app.get('/health')
def health_check():
    try:
        summary = dataset_summary()
        return {
            'status': 'ok',
            'dataset_loaded': True,
            'dataset_files': summary['dataset_files'],
            'records_loaded': summary['records_loaded'],
            'unique_banks': summary['unique_banks'],
            'months_loaded': summary['months_loaded'],
            'message': 'Excel dataset loaded successfully.',
        }
    except Exception as exc:
        logger.exception('Health check failed while loading dataset.')
        dataset_files = 0
        try:
            dataset_files = len(get_dataset_files())
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load the Excel dataset. Files found: {dataset_files}. Error: {exc}",
        ) from exc


@app.get('/api/dashboard')
def get_dashboard():
    try:
        df = load_clean_dataset()
        monthly = df.groupby('month_label', as_index=False)['cash_requirement_inr'].sum()
        monthly = monthly.sort_values('month_label')
        total_unique_banks = df['bank_name'].nunique()
        total_records = len(df)
        avg_requirement = float(df['cash_requirement_inr'].mean())
        peak_month = monthly.iloc[monthly['cash_requirement_inr'].idxmax()]

        return {
            'summary': {
                'total_records': total_records,
                'unique_banks': total_unique_banks,
                'average_cash_requirement_inr': avg_requirement,
                'peak_month': peak_month['month_label'],
                'peak_cash_requirement_inr': float(peak_month['cash_requirement_inr']),
            },
            'historical_trend': [
                {'month': row['month_label'], 'cash_requirement_inr': float(row['cash_requirement_inr'])}
                for _, row in monthly.iterrows()
            ],
            'bank_count': total_unique_banks,
        }
    except Exception as exc:
        logger.exception('Failed to build dashboard data from the Excel source.')
        raise HTTPException(status_code=500, detail=f'Dashboard failed to load real data: {exc}') from exc


@app.get('/api/feature-options')
def feature_options():
    try:
        df = load_clean_dataset()
        return {
            'banks': sorted(df['bank_name'].dropna().unique().tolist()),
            'months': sorted(df['month_label'].dropna().unique().tolist()),
            'target_column': 'cash_requirement_inr',
        }
    except Exception as exc:
        logger.exception('Failed to load feature options from the dataset.')
        raise HTTPException(status_code=500, detail=f'Feature options unavailable: {exc}') from exc


@app.get('/api/data-audit')
def data_audit():
    try:
        df = load_clean_dataset()
        files = get_dataset_files()
        date_min = df['date'].min() if 'date' in df.columns else None
        date_max = df['date'].max() if 'date' in df.columns else None
        required_columns = {
            'atm_identifier': 'atm_id' in df.columns or 'atm_number' in df.columns or 'atm_name' in df.columns,
            'daily_date': 'daily_date' in df.columns or 'date' in df.columns and df['date'].dt.day.nunique() > 1,
            'daily_atm_cash_history': 'cash_withdrawal_at_atm_value' in df.columns and 'bank_name' in df.columns and 'date' in df.columns,
            'holiday_calendar': 'holiday_flag' in df.columns or 'is_holiday' in df.columns,
        }
        missing_data = {
            'atm_identifier': not required_columns['atm_identifier'],
            'daily_atm_history': not required_columns['daily_atm_cash_history'],
            'holiday_calendar': not required_columns['holiday_calendar'],
            'daily_prediction_target': not required_columns['daily_date'],
        }
        return {
            'dataset_kind': 'monthly_bank_level',
            'observation_frequency': 'monthly',
            'source_files': len(files),
            'records_loaded': int(len(df)),
            'unique_banks': int(df['bank_name'].nunique()),
            'date_range': {
                'start': date_min.strftime('%Y-%m-%d') if date_min is not None else None,
                'end': date_max.strftime('%Y-%m-%d') if date_max is not None else None,
            },
            'available_columns': list(df.columns),
            'supports_daily_atm_forecast': False,
            'missing_data': missing_data,
            'message': 'The source files are monthly bank aggregates; they do not contain ATM-level daily cash withdrawal histories or ATM identifiers needed for genuine ATM-day forecasting.',
        }
    except Exception as exc:
        logger.exception('Dataset audit failed.')
        raise HTTPException(status_code=500, detail=f'Dataset audit unavailable: {exc}') from exc


@app.get('/api/daily-capability')
def daily_capability():
    try:
        df = load_synthetic_daily_dataset()
        return {
            'supports_daily_atm_forecast': True,
            'dataset_kind': 'synthetic_daily_demo',
            'observation_frequency': 'daily',
            'message': 'Synthetic Demo Prediction — Trained on 2023 Data. The model learns from 2023-only synthetic observations and can generate future-date features without requiring an exact historical match.',
            'missing_data': {
                'atm_identifier': False,
                'daily_atm_history': False,
                'holiday_calendar': False,
                'daily_prediction_target': False,
            },
            'available_columns': list(df.columns),
            'atm_ids': sorted(df['ATM_ID'].dropna().astype(str).unique().tolist()),
            'date_range': {
                'start': '2023-01-01',
                'end': '2035-12-31',
            },
            'currency_unit': 'synthetic demo units',
        }
    except Exception as exc:
        logger.warning('Synthetic daily ATM capability check failed: %s', exc)
        return {
            'supports_daily_atm_forecast': False,
            'dataset_kind': 'monthly_bank_level',
            'observation_frequency': 'monthly',
            'message': 'Synthetic daily ATM data is unavailable. The project remains on monthly RBI bank data for real analysis.',
            'missing_data': {
                'atm_identifier': True,
                'daily_atm_history': True,
                'holiday_calendar': True,
                'daily_prediction_target': True,
            },
            'available_columns': [],
            'atm_ids': [],
            'date_range': {'start': None, 'end': None},
            'currency_unit': 'INR (demo unavailable)',
        }


@app.get('/api/trend-analysis')
def trend_analysis(
    bank_name: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
):
    try:
        df = load_clean_dataset().copy()
        if 'month_label' not in df.columns or 'cash_requirement_inr' not in df.columns:
            raise ValueError('The dataset is missing the monthly bank-level cash requirement fields required for trend analysis.')

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        month_options = sorted(
            df['month_label'].dropna().unique().tolist(),
            key=lambda value: pd.to_datetime(value, format='%b %Y'),
        )
        bank_options = sorted(df['bank_name'].dropna().unique().tolist())

        if start_month and start_month not in month_options:
            raise HTTPException(status_code=400, detail=f'Unknown start_month: {start_month}')
        if end_month and end_month not in month_options:
            raise HTTPException(status_code=400, detail=f'Unknown end_month: {end_month}')
        if start_month and end_month and month_options.index(start_month) > month_options.index(end_month):
            raise HTTPException(status_code=400, detail='start_month must be earlier than or equal to end_month.')

        selected_bank = bank_name or 'All Banks'
        if selected_bank != 'All Banks' and selected_bank not in bank_options:
            raise HTTPException(status_code=404, detail=f'Bank not found: {selected_bank}')

        filtered = df.copy()
        if selected_bank != 'All Banks':
            filtered = filtered[filtered['bank_name'] == selected_bank].copy()

        if start_month:
            filtered = filtered[filtered['month_label'] >= start_month].copy()
        if end_month:
            filtered = filtered[filtered['month_label'] <= end_month].copy()

        monthly = (
            filtered.groupby(['month_label', 'date'], as_index=False)
            .agg(
                cash_requirement_inr=('cash_requirement_inr', 'sum'),
                cash_withdrawal_total_atm_volume=('cash_withdrawal_total_atm_volume', 'sum'),
            )
            .sort_values('date')
            .reset_index(drop=True)
        )
        monthly['month'] = monthly['month_label']
        monthly['cash_requirement_inr'] = monthly['cash_requirement_inr'].fillna(0)
        monthly['cash_withdrawal_total_atm_volume'] = monthly['cash_withdrawal_total_atm_volume'].fillna(0)

        summary = {
            'total_cash_requirement_inr': float(monthly['cash_requirement_inr'].sum()),
            'total_cash_withdrawal_volume': float(monthly['cash_withdrawal_total_atm_volume'].sum()),
            'peak_month': None,
            'peak_cash_requirement_inr': None,
            'percentage_change_vs_previous_month': None,
        }

        if not monthly.empty:
            peak_row = monthly.loc[monthly['cash_requirement_inr'].idxmax()]
            summary['peak_month'] = peak_row['month_label']
            summary['peak_cash_requirement_inr'] = float(peak_row['cash_requirement_inr'])

            if len(monthly) > 1:
                current_value = float(monthly['cash_requirement_inr'].iloc[-1])
                previous_value = float(monthly['cash_requirement_inr'].iloc[-2])
                if previous_value not in (0, None):
                    summary['percentage_change_vs_previous_month'] = float(((current_value - previous_value) / previous_value) * 100)
                else:
                    summary['percentage_change_vs_previous_month'] = None

        return {
            'selected_bank': selected_bank,
            'available_banks': bank_options,
            'available_months': month_options,
            'summary': summary,
            'series': [
                {
                    'month': row['month_label'],
                    'month_label': row['month_label'],
                    'cash_requirement_inr': float(row['cash_requirement_inr']),
                    'cash_withdrawal_total_atm_volume': float(row['cash_withdrawal_total_atm_volume']),
                }
                for _, row in monthly.iterrows()
            ],
            'date_range': {
                'start_month': month_options[0] if month_options else None,
                'end_month': month_options[-1] if month_options else None,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Failed to generate monthly trend analysis.')
        raise HTTPException(status_code=500, detail=f'Trend analysis failed: {exc}') from exc


@app.get('/api/analytics')
def analytics():
    try:
        df = load_clean_dataset()
        by_bank = df.groupby('bank_name', as_index=False)['cash_requirement_inr'].mean().sort_values('cash_requirement_inr', ascending=False)
        by_month = df.groupby('month_label', as_index=False)['cash_requirement_inr'].sum().sort_values('month_label')

        return {
            'top_banks': [
                {'bank_name': row['bank_name'], 'average_cash_requirement_inr': float(row['cash_requirement_inr'])}
                for _, row in by_bank.head(10).iterrows()
            ],
            'monthly_totals': [
                {'month': row['month_label'], 'cash_requirement_inr': float(row['cash_requirement_inr'])}
                for _, row in by_month.iterrows()
            ],
        }
    except Exception as exc:
        logger.exception('Failed to generate analytics from the dataset.')
        raise HTTPException(status_code=500, detail=f'Analytics failed: {exc}') from exc


@app.get('/api/performance')
def performance():
    try:
        metrics = train_model_and_metrics()
        data = []
        for actual, predicted in zip(metrics['actual'], metrics['predicted']):
            data.append({'actual': float(actual), 'predicted': float(predicted)})
        return {
            'model_name': metrics['model_name'],
            'metrics': metrics['metrics'],
            'comparison': data,
        }
    except Exception as exc:
        logger.exception('Failed to train or evaluate the model.')
        raise HTTPException(status_code=500, detail=f'Model performance unavailable: {exc}') from exc


@app.get('/api/history')
def prediction_history():
    if not HISTORY_PATH.exists():
        return {'records': []}
    records = json.loads(HISTORY_PATH.read_text(encoding='utf-8') or '[]')
    return {'records': records}


@app.post('/api/predict')
def predict(payload: PredictionPayload):
    try:
        payload_dict = payload.model_dump(exclude_none=True)
        result = predict_cash_requirement(payload_dict)
        record = {
            'bank_name': result['bank_name'],
            'month': result['month'],
            'cash_requirement_inr': float(result['cash_requirement_inr']),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        append_prediction_record(record)
        return {
            'prediction': {
                'bank_name': result['bank_name'],
                'month': result['month'],
                'cash_requirement_inr': float(result['cash_requirement_inr']),
                'cash_requirement_value': float(result['cash_requirement_value']),
            },
            'status': 'success',
        }
    except Exception as exc:
        logger.exception('Prediction request failed.')
        raise HTTPException(status_code=500, detail=f'Prediction failed: {exc}') from exc


@app.post('/api/predict-daily')
async def predict_daily(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Request body must be valid JSON.') from exc

    capability = daily_capability()
    if not capability['supports_daily_atm_forecast']:
        raise HTTPException(
            status_code=400,
            detail=(
                'Daily ATM forecasting is not supported with the current dataset. '
                'The project remains on monthly RBI bank data unless a daily ATM dataset is added.'
            ),
        )

    atm_id = payload.get('atm_id')
    prediction_date = payload.get('prediction_date')
    if not atm_id or not prediction_date:
        raise HTTPException(status_code=400, detail='Both atm_id and prediction_date are required.')

    try:
        result = predict_daily_atm_cash({'atm_id': atm_id, 'prediction_date': prediction_date})
        return {
            'status': 'success',
            'prediction': result,
            'dataset_kind': 'synthetic_daily_demo',
            'currency_unit': 'INR (synthetic demo)',
            'note': 'This is a synthetic daily ATM forecast for demonstration purposes only and should not be treated as real-world RBI data.',
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Synthetic daily ATM prediction failed.')
        raise HTTPException(status_code=500, detail=f'Synthetic daily ATM prediction failed: {exc}') from exc


@app.get('/')
def root():
    return {'message': 'ATM Cash Requirement Prediction API is running.'}
