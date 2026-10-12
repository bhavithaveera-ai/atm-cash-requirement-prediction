from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class PredictionPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    bank_name: str = Field(..., min_length=2)
    month: str = Field(...)
    atm_on_site: float = 0.0
    atm_off_site: float = 0.0
    pos_count: float = 0.0
    micro_atm_count: float = 0.0
    bharat_qr_codes: float = 0.0
    upi_qr_codes: float = 0.0
    credit_cards: float = 0.0
    debit_cards: float = 0.0
    card_payments_pos_volume: float = 0.0
    card_payments_pos_value: float = 0.0
    card_payments_online_volume: float = 0.0
    card_payments_online_value: float = 0.0
    card_payments_others_volume: float = 0.0
    card_payments_others_value: float = 0.0
    cash_withdrawal_at_atm_volume: float = 0.0
    cash_withdrawal_at_atm_value: float = 0.0
    cash_withdrawal_at_pos_volume: float = 0.0
    cash_withdrawal_at_pos_value: float = 0.0
    cash_withdrawal_online_volume: float = 0.0
    cash_withdrawal_online_value: float = 0.0
    cash_withdrawal_others_volume: float = 0.0
    cash_withdrawal_others_value: float = 0.0
    cash_withdrawal_total_atm_volume: float = 0.0
    cash_withdrawal_total_atm_value: float = 0.0
    cash_withdrawal_total_pos_volume: float = 0.0
    cash_withdrawal_total_pos_value: float = 0.0


class DailyPredictionPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    atm_id: str = Field(..., min_length=2)
    prediction_date: str = Field(...)
    available_cash_balance: float = 0.0


class PredictionHistoryRecord(BaseModel):
    bank_name: str
    month: str
    cash_requirement_inr: float
    created_at: str
