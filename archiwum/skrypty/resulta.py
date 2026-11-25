# --- ARCHIWUM (praca inżynierska) – uruchamiaj z głównego folderu: python archiwum/skrypty/<plik>.py
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_sys.path.insert(0, _ROOT)                                   # data_loader z głównego folderu
_os.makedirs(_os.path.join(_ROOT, "archiwum", "wyniki"), exist_ok=True)
_os.chdir(_os.path.join(_ROOT, "archiwum", "wyniki"))        # wykresy/CSV zapisują się w archiwum/wyniki
# ---
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_loader import load_data

# >>> DODANE: SARIMA (AutoARIMA)
from pmdarima import auto_arima


# ==========================================
# 1. INŻYNIERIA CECH (the same as in the forecast)
# ==========================================
def create_features(df):
    df = df.copy()
    df['lag_1'] = df['CPI'].shift(1)
    df['lag_2'] = df['CPI'].shift(2)
    df['lag_3'] = df['CPI'].shift(3)
    df['lag_6'] = df['CPI'].shift(6)
    df['lag_12'] = df['CPI'].shift(12)

    df['roll_mean_3'] = df['CPI'].shift(1).rolling(window=3).mean()
    df['roll_mean_6'] = df['CPI'].shift(1).rolling(window=6).mean()

    df['month'] = df.index.month
    df['quarter'] = df.index.quarter

    df = df.dropna()
    return df


# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
df_raw = load_data()

# (opcjonalnie, ale zalecane) upewnij się, że index to DatetimeIndex
df_raw = df_raw.copy()
df_raw.index = pd.to_datetime(df_raw.index)

df_features = create_features(df_raw)

features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'roll_mean_3', 'roll_mean_6', 'month', 'quarter']
target = 'CPI'

X = df_features[features]
y = df_features[target]

# Podział na Train / Test
split_date = pd.Timestamp("2023-01-01")

X_train = X.loc[X.index < split_date]
y_train = y.loc[y.index < split_date]
X_test  = X.loc[X.index >= split_date]
y_test  = y.loc[y.index >= split_date]

print(f">>> Train: {len(X_train)} | Test: {len(X_test)}")


# ==========================================
# 3. MODELOWANIE I OCENA
# ==========================================

# A. XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# B. Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# C. SARIMA (AutoARIMA) - trenowana tylko na y_train, prognoza na długość testu
# Uwaga: m=12 zakłada sezonowość roczną przy danych miesięcznych
sarima_model = auto_arima(
    y_train,
    seasonal=True,
    m=12,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=False
)
sarima_forecast = sarima_model.predict(n_periods=len(y_test))
sarima_pred = pd.Series(sarima_forecast, index=y_test.index)


def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n--- {name} ---")
    print(f"   MAE : {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    return mae, rmse


print("\n>>> WYNIKI WALIDACJI (Ewaluacja modeli):")
mae_xgb, rmse_xgb = print_metrics("XGBoost", y_test, xgb_pred)
mae_rf, rmse_rf = print_metrics("Random Forest", y_test, rf_pred)
mae_sarima, rmse_sarima = print_metrics("SARIMA (AutoARIMA)", y_test, sarima_pred)


# ==========================================
# 4. WIZUALIZACJA 1: PORÓWNANIE PROGNOZ (Test set)
# ==========================================
plt.figure(figsize=(12, 6))

plot_start = "2020-01-01"
train_subset = y_train.loc[plot_start:]

plt.plot(train_subset.index, train_subset, label="Train (Historia)", color='gray', alpha=0.5)
plt.plot(y_test.index, y_test, label="Test (Rzeczywiste)", color='black', linewidth=2)

plt.plot(y_test.index, xgb_pred, label=f"XGBoost (MAE={mae_xgb:.2f})", color='red', linestyle='--')
plt.plot(y_test.index, rf_pred, label=f"Random Forest (MAE={mae_rf:.2f})", color='#DAA520', linestyle='--')
plt.plot(y_test.index, sarima_pred, label=f"SARIMA (MAE={mae_sarima:.2f})", color='blue', linestyle='--')

plt.title("Walidacja modeli: prognoza w reżimie dezinflacji (test od 2023)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ml_forecast_comparison.png")
print(">>> Zapisano wykres porównawczy: ml_forecast_comparison.png")
plt.close()


# ==========================================
# 5. WIZUALIZACJA 2: WAŻNOŚĆ CECH (XGBoost)
# ==========================================
plt.figure(figsize=(10, 6))
importances = xgb_model.feature_importances_
indices = np.argsort(importances)

plt.title('Ważność cech wg XGBoost')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Waga cechy')
plt.tight_layout()
plt.savefig("feature_importance.png")
print(">>> Zapisano ważność cech: feature_importance.png")
plt.close()
