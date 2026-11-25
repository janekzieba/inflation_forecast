import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from app import get_inflation_status, calculate_metrics, load_data_app, calculate_return_date, generate_pdf_report


# ==========================================
# 1. TESTY JEDNOSTKOWE (Unit Tests)
# ==========================================

def test_inflation_status_logic():
    """Sprawdza kolory i statusy inflacji."""
    # 1. Deflacja
    status, color = get_inflation_status(99.0)
    assert "DEFLACJA" in status
    assert color == "#3498db"
    # 2. Cel NBP
    status, color = get_inflation_status(102.5)
    assert "W CELU NBP" in status
    assert color == "#27ae60"
    # 3. Kryzys
    status, color = get_inflation_status(110.0)
    assert "KRYZYSOWA" in status


def test_metrics_calculation():
    """Sprawdza matematykę błędów (MAE)."""
    df_test = pd.DataFrame({
        'Actual': [100, 102, 104],
        'XGBoost': [100, 102, 104],  # Idealne (MAE 0)
        'RandomForest': [101, 103, 105],  # Błąd +1
        'SARIMA': [100, 102, 104],
        'Naive': [100, 102, 104]
    })
    results = calculate_metrics(df_test)

    xgb_row = results[results['Model'] == 'XGBoost'].iloc[0]
    assert xgb_row['MAE'] == 0.0

    rf_row = results[results['Model'] == 'RandomForest'].iloc[0]
    assert rf_row['MAE'] == 1.0


def test_return_to_target_logic():
    """Sprawdza, czy system dobrze wykrywa datę powrotu do celu NBP."""
    # Scenariusz A: Inflacja spada do celu (102.0 jest w przedziale 101.5-103.5)
    dates = pd.date_range("2025-01-01", periods=3, freq="MS")
    # Seria: 105 (Wysoka), 104 (Wysoka), 102 (Cel)
    series_ok = pd.Series([105.0, 104.0, 102.0], index=dates)

    result_date = calculate_return_date(series_ok)
    assert result_date == pd.Timestamp("2025-03-01")

    # Scenariusz B: Inflacja nigdy nie wraca do celu
    series_bad = pd.Series([110.0, 109.0, 108.0], index=dates)
    assert calculate_return_date(series_bad) is None


def test_pdf_generator():
    """Sprawdza, czy generator PDF nie wyrzuca błędów i zwraca bajty."""
    # Generujemy przykładowy PDF
    pdf_bytes = generate_pdf_report("2025-10", 105.5, "Wysoka", "XGBoost", 900.0)

    # Sprawdzamy czy to są dane binarne (bytes) i czy nie jest puste
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    # Sprawdzamy czy PDF zaczyna się od nagłówka PDF (magiczne bajty)
    assert pdf_bytes.startswith(b"%PDF")


def test_data_integrity():
    """Sprawdza, czy pliki CSV istnieją i mają poprawne kolumny."""
    #Prawdziwe pliki z dysku
    hist, sarima, ml, naive, val, lstm = load_data_app()

    # Czy dane nie są puste?
    assert not hist.empty
    assert not ml.empty

    # Czy kluczowe kolumny istnieją?
    assert 'CPI' in hist.columns
    assert 'CPI_forecast_XGB' in ml.columns
    assert 'CPI_forecast' in sarima.columns
    # Moduł AI (LSTM): prognoza na 12 mies. i prognozy testowe w zakładce ewaluacji
    assert lstm is not None and len(lstm) == 12
    assert {'LSTM', 'LSTM_lo_conf', 'LSTM_hi_conf'} <= set(lstm.columns)
    assert 'LSTM' in val.columns


# ==========================================
# 2. TESTY INTEGRACYJNE (Integration Tests)
# ==========================================

def test_app_startup():
    """Smoke Test: Czy aplikacja wstaje."""
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    assert not at.exception
    assert "System Predykcji" in at.title[0].value


def test_sidebar_interaction():
    """Czy zmiana suwaka szoku działa."""
    at = AppTest.from_file("app.py")
    at.run(timeout=60)
    at.sidebar.slider(key="shock").set_value(2.0).run(timeout=30)
    assert len(at.warning) > 0


def test_tabs_navigation():
    """Sprawdza zakładki: 3 główne + 5 podzakładek modułu LSTM & XAI."""
    at = AppTest.from_file("app.py")
    at.run(timeout=60)

    labels = [t.label for t in at.tabs]
    assert len(at.tabs) == 8
    assert "Dashboard" in labels[0]
    assert any("LSTM & XAI" in l for l in labels)
    assert any("Ocena końcowa" in l for l in labels)


def test_lstm_module_outputs():
    """Czy wyniki modułu AI potrzebne aplikacji są na miejscu."""
    from pathlib import Path
    for f in ["results/metrics_summary.csv", "results/dm_tests.csv", "results/forecasts.csv",
              "results/xai/importance.csv", "results/final/holdout_metrics.csv", "lstm_forecast_explanation.csv"]:
        assert Path(f).exists(), f