import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from data_loader import load_data


# ==========================================
# 1. DEFINICJE FUNKCJI
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
    return df


# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
df_full = load_data()
df_features = create_features(df_full)
df_train = df_features.dropna()

features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'roll_mean_3', 'roll_mean_6', 'month', 'quarter']
target = 'CPI'

X_train = df_train[features]
y_train = df_train[target]

print(f">>> Trenowanie modeli na {len(y_train)} rekordach...")

# --- MODEL 1: XGBoost ---
model_xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=42, n_jobs=-1
)
model_xgb.fit(X_train, y_train)
print(">>> XGBoost wytrenowany.")

# --- MODEL 2: Random Forest (NOWOŚĆ) ---
model_rf = RandomForestRegressor(
    n_estimators=500, max_depth=10, random_state=42, n_jobs=-1
)
model_rf.fit(X_train, y_train)
print(">>> Random Forest wytrenowany.")

# ==========================================
# 3. PROGNOZA REKURENCYJNA (Dla obu modeli)
# ==========================================
horizon_months = 120
future_preds_xgb = []
future_preds_rf = []

# 2 kopie danych, dla kazdego modelu
current_df_xgb = df_full.copy()
current_df_rf = df_full.copy()

print(f">>> Generowanie prognoz na {horizon_months} miesięcy...")

for step in range(horizon_months):
    # --- Ścieżka XGBoost ---
    feats_xgb = create_features(current_df_xgb)
    last_date = current_df_xgb.index[-1]
    next_date = last_date + pd.DateOffset(months=1)

    # Pusty wiersz dla lagow
    next_row = pd.DataFrame({'CPI': [np.nan]}, index=[next_date])
    temp_df_xgb = pd.concat([current_df_xgb, next_row])
    temp_feats_xgb = create_features(temp_df_xgb)
    X_next_xgb = temp_feats_xgb.iloc[[-1]][features]

    pred_val_xgb = model_xgb.predict(X_next_xgb)[0]
    future_preds_xgb.append(pred_val_xgb)

    # Aktualizacja historii XGB
    real_next_row_xgb = pd.DataFrame({'CPI': [pred_val_xgb]}, index=[next_date])
    current_df_xgb = pd.concat([current_df_xgb, real_next_row_xgb])

    # --- Ścieżka Random Forest ---
    feats_rf = create_features(current_df_rf)

    temp_df_rf = pd.concat([current_df_rf, next_row])  # next_row jest taki sam (pusty)
    temp_feats_rf = create_features(temp_df_rf)
    X_next_rf = temp_feats_rf.iloc[[-1]][features]

    pred_val_rf = model_rf.predict(X_next_rf)[0]
    future_preds_rf.append(pred_val_rf)

    # Aktualizacja historii RF
    real_next_row_rf = pd.DataFrame({'CPI': [pred_val_rf]}, index=[next_date])
    current_df_rf = pd.concat([current_df_rf, real_next_row_rf])

# ==========================================
# 4. ZAPIS
# ==========================================
forecast_index = pd.date_range(
    start=df_full.index[-1] + pd.DateOffset(months=1),
    periods=horizon_months, freq='MS'
)

forecast_df = pd.DataFrame({
    'date': forecast_index,
    'CPI_forecast_XGB': future_preds_xgb,
    'CPI_forecast_RF': future_preds_rf
})

forecast_df.to_csv("ml_10y_forecast_combined.csv", index=False)
print(">>> Zapisano prognozy (XGB + RF) do: ml_10y_forecast_combined.csv")