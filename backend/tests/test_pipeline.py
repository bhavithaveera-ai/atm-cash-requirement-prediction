import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.data_loader import load_clean_dataset
from backend.app.main import app, feature_options, get_dashboard, health_check
from backend.app.model_service import (
    predict_daily_atm_cash,
    train_model_and_metrics,
    train_synthetic_daily_model_and_metrics,
)

client = TestClient(app)


def test_clean_dataset_is_loaded_from_raw_files():
    df = load_clean_dataset()

    assert not df.empty
    assert 'bank_name' in df.columns
    assert 'month' in df.columns
    assert 'cash_requirement_value' in df.columns
    assert df['cash_requirement_value'].gt(0).all()


def test_model_training_returns_metrics():
    result = train_model_and_metrics()

    assert 'model_name' in result
    assert 'metrics' in result
    assert set(['mae', 'rmse', 'r2_score']).issubset(result['metrics'])
    assert result['metrics']['r2_score'] <= 1.0


def test_health_and_dashboard_endpoints_report_real_data():
    health_data = health_check()
    assert health_data['status'] == 'ok'
    assert health_data['dataset_loaded'] is True
    assert health_data['records_loaded'] > 0

    dashboard_data = get_dashboard()
    assert dashboard_data['summary']['total_records'] > 0
    assert dashboard_data['summary']['unique_banks'] > 0

    feature_payload = feature_options()
    assert len(feature_payload['banks']) > 0
    assert len(feature_payload['months']) > 0


def test_synthetic_daily_dataset_is_available_and_well_formed():
    from pathlib import Path

    dataset_path = Path('datasets/synthetic_atm_daily_data.csv')
    assert dataset_path.exists(), 'Synthetic daily ATM dataset was not copied into the datasets directory.'

    df = pd.read_csv(dataset_path)
    assert not df.empty
    assert {'Date', 'ATM_ID', 'Location', 'Withdrawals', 'DayOfWeek', 'IsHoliday', 'WeatherImpact'}.issubset(df.columns)
    assert df.duplicated().sum() == 0
    assert df['Withdrawals'].notna().all()
    assert df['Date'].min() < df['Date'].max()


def test_daily_model_trains_only_on_2023_rows():
    df = pd.read_csv('datasets/synthetic_atm_daily_data.csv', parse_dates=['Date'])
    df_2023 = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2023-12-31')].copy()

    assert len(df_2023) == 1824
    assert df_2023['ATM_ID'].nunique() == 5
    assert df_2023['Date'].min().strftime('%Y-%m-%d') == '2023-01-01'
    assert df_2023['Date'].max().strftime('%Y-%m-%d') == '2023-12-31'

    summary = train_synthetic_daily_model_and_metrics()
    assert summary['dataset_rows'] == len(df_2023)
    assert summary['date_range']['start'] == '2023-01-01'
    assert summary['date_range']['end'] == '2023-12-31'
    assert summary['train_rows'] > 0
    assert summary['test_rows'] > 0


def test_daily_prediction_supports_future_dates_outside_2023():
    prediction = predict_daily_atm_cash({'atm_id': 'ATM_1', 'prediction_date': '2024-02-15'})
    assert prediction['atm_id'] == 'ATM_1'
    assert prediction['prediction_date'] == '2024-02-15'
    assert prediction['predicted_cash_requirement_inr'] > 0

    prediction_2025 = predict_daily_atm_cash({'atm_id': 'ATM_3', 'prediction_date': '2025-01-15'})
    assert prediction_2025['prediction_date'] == '2025-01-15'
    assert prediction_2025['predicted_cash_requirement_inr'] > 0

    with pytest.raises(ValueError):
        predict_daily_atm_cash({'atm_id': 'ATM_999', 'prediction_date': '2024-02-15'})


def test_daily_prediction_endpoint_accepts_synthetic_daily_model_when_available():
    response = client.post(
        '/api/predict-daily',
        json={
            'atm_id': 'ATM_1',
            'prediction_date': '2025-01-01',
            'available_cash_balance': 50000,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'success'
    assert payload['prediction']['atm_id'] == 'ATM_1'
    assert payload['prediction']['predicted_cash_requirement_inr'] > 0


def test_daily_prediction_capability_endpoint_reports_gap_or_support_for_synthetic_dataset():
    response = client.get('/api/daily-capability')
    assert response.status_code == 200
    payload = response.json()
    assert 'supports_daily_atm_forecast' in payload
    assert payload['supports_daily_atm_forecast'] in {True, False}


def test_data_audit_reports_monthly_dataset_honestly_and_keeps_daily_demo_separate():
    response = client.get('/api/data-audit')
    assert response.status_code == 200
    payload = response.json()

    assert payload['missing_data']['atm_identifier'] is True
    assert payload['missing_data']['daily_atm_history'] is True
    assert payload['missing_data']['holiday_calendar'] is True
    assert payload['missing_data']['daily_prediction_target'] is True
    assert payload['monthly_dataset']['date_range']['start'] is not None
    assert payload['synthetic_daily_dataset']['supports_daily_atm_forecast'] is True
    assert payload['synthetic_daily_dataset']['missing_data']['atm_identifier'] is False


def test_trend_analysis_endpoint_returns_monthly_bank_data():
    response = client.get('/api/trend-analysis?bank_name=All+Banks')
    assert response.status_code == 200
    payload = response.json()
    assert payload['selected_bank'] == 'All Banks'
    assert 'series' in payload
    assert len(payload['series']) > 0
    assert 'summary' in payload
    assert payload['summary']['total_cash_requirement_inr'] > 0

    bank_response = client.get('/api/trend-analysis?bank_name=BANK+OF+BARODA')
    assert bank_response.status_code == 200
    bank_payload = bank_response.json()
    assert bank_payload['selected_bank'] == 'BANK OF BARODA'
    assert bank_payload['summary']['total_cash_requirement_inr'] > 0
