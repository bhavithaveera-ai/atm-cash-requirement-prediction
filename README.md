# ATM Cash Requirement Prediction Using Machine Learning

This project uses the original Excel files in the `datasets/` folder to train and evaluate a realistic cash-demand model for the available bank-level monthly data, while clearly separating that from the unsupported daily ATM-level prediction use case.

## Dataset audit and limitations

The real source files are 10 monthly Excel workbooks under `datasets/`.

### What the source data actually contains

Each workbook is a monthly bank summary. The source rows represent banks for a given month, not individual ATMs. The data includes:

- `bank_name`
- `Sr. No.`
- `ATMs and CRMs` - on-site count, off-site count
- `PoS` count
- `Micro ATMs`
- `Bharat QR codes`
- `UPI QR codes`
- `Credit cards`
- `Debit cards`
- `Card Payments Transactions at PoS`, `Online`, and `Others`
- `Cash Withdrawal at ATM`, `at PoS`, `Online`, `Others`
- `Total ATM cash withdrawal volume` and `Total ATM cash withdrawal value`
- `Total POS transaction value`

These values are reported as monthly bank-level statistics, not day-by-day ATM history.

### What is missing for individual ATM daily forecasting

The current dataset does not include any of the following:

- ATM identifiers or ATM numbers per record
- Daily ATM-level cash withdrawals
- Date-level observation for each ATM
- Day-of-week / weekend flags
- Public holiday / pre-holiday / post-holiday flags
- Previous-day or rolling demand history per ATM
- Historical demand for a single ATM across dates

Because of this, a genuine daily cash prediction for an individual ATM cannot be trained or inferred from the current files without fabricating an ATM-level dataset.

### Target variable used in the supported model

The supported regression model uses the bank-level monthly target:

- `cash_withdrawal_total_atm_value`

This is converted from the source `Value (in Rs'000)` scale to rupee units in the backend and is used as the prediction target for the monthly bank-level dashboard model.

### Included and excluded features

Included features for the supported bank-level monthly model:

- `atm_on_site`
- `atm_off_site`
- `pos_count`
- `micro_atm_count`
- `bharat_qr_codes`
- `upi_qr_codes`
- `credit_cards`
- `debit_cards`
- `card_payments_pos_value`
- `card_payments_online_value`
- `cash_withdrawal_total_atm_volume`
- `cash_withdrawal_total_pos_value`
- `bank_name`, `month_number`, `year`

Excluded because they are absent, not ATM-specific, or not valid at prediction time:

- ATM-level identifiers and ATM names
- Daily transaction data
- Daily holiday or calendar features
- Past daily demand lags for a single ATM
- Future realized withdrawal amounts used as features
- Synthetic daily values created by dividing monthly totals by day counts

### Supported model

The current working model is a `RandomForestRegressor` trained on the cleaned monthly bank-level dataset using chronological hold-out validation, with MAE, RMSE, and R² reported in the API.

The application intentionally keeps this monthly forecast separate from the unsupported daily ATM forecast. The backend explicitly refuses to fabricate a daily prediction when the dataset does not support it.

## Project structure

- `datasets/` – original Excel datasets. Do not modify these files.
- `backend/` – FastAPI app, data loader, model service, and API endpoints.
- `data/processed/` – cleaned CSV export and prediction history.
- `models/` – trained model artifact.
- `frontend/` – Vite + React dashboard.

## Python environment

Use the verified working environment:

```powershell
C:\venvs\ATMPROJ312\Scripts\python.exe
```

## Install backend dependencies

```powershell
cd "C:\Users\bhavi\OneDrive\Desktop\atmcashrequiremnet"
C:\venvs\ATMPROJ312\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\ATMPROJ312\Scripts\python.exe -m pip install fastapi uvicorn pandas numpy scikit-learn openpyxl joblib python-multipart pytest httpx
```

## Install frontend dependencies

```powershell
cd "C:\Users\bhavi\OneDrive\Desktop\atmcashrequiremnet\frontend"
npm install
```

## Start the backend

```powershell
cd "C:\Users\bhavi\OneDrive\Desktop\atmcashrequiremnet"
C:\venvs\ATMPROJ312\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Start the frontend

```powershell
cd "C:\Users\bhavi\OneDrive\Desktop\atmcashrequiremnet\frontend"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
npm run dev -- --host 0.0.0.0 --port 5173
```

## Run tests

```powershell
cd "C:\Users\bhavi\OneDrive\Desktop\atmcashrequiremnet"
C:\venvs\ATMPROJ312\Scripts\python.exe -m pytest backend/tests -q
```

## API endpoints

- `GET /health`
- `GET /api/dashboard`
- `GET /api/feature-options`
- `GET /api/analytics`
- `GET /api/performance`
- `GET /api/history`
- `GET /api/daily-capability`
- `POST /api/predict`
- `POST /api/predict-daily`

## Important notes

- The original raw Excel files in `datasets/` are never edited, renamed, or overwritten.
- The dashboard and model operate only on the real source data and the cleaned export generated from it.
- The project explicitly reports when a requested daily ATM forecast is unsupported instead of returning fake values.
