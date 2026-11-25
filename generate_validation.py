import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from data_loader import load_data

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

print(">>> Generowanie danych walidacyjnych (2015-2025)... to potrwa chwilę.")

# 1. Dane
df = load_data()
df_feat = create_features(df).dropna()

# Ustalamy okres walidacji: od 2015 do teraz
val_start_date = "2015-01-01"
val_data = df_feat.loc[val_start_date:].copy()

# Przygotowanie zbiorów
X_full = df_feat.drop(columns=['CPI'])
y_full = df_feat['CPI']

# --- MODELE ML (Cross-Validation style: Trenujemy na danych DO 2015 i przewidujemy resztę) ---
# Trenujemy na danych dostępnych przed okresem testowym. (zamiast okna przesuwnego)
split_idx = df_feat.index.get_loc(val_start_date)
X_train = X_full.iloc[:split_idx]
y_train = y_full.iloc[:split_idx]

# XGBoost
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
# Przewidujemy na zbiorze walidacyjnym (nieznanym modelowi)
val_data['Pred_XGBoost'] = xgb.predict(X_full.loc[val_start_date:])

# Random Forest
rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
val_data['Pred_RandomForest'] = rf.predict(X_full.loc[val_start_date:])

# --- SARIMA (In-sample prediction) ---
# Trenujemy na całości, ale bierzemy fitted values (dopasowanie historyczne)
sarima = SARIMAX(df['CPI'], order=(2, 0, 1), seasonal_order=(2, 1, 0, 12), enforce_stationarity=False).fit(disp=False)
pred_sarima = sarima.predict(start=val_start_date, end=df.index[-1])
val_data['Pred_SARIMA'] = pred_sarima

# --- NAIVE (Benchmark) ---
# Seasonal Naive: Wartość sprzed 12 miesięcy
val_data['Pred_Naive'] = df['CPI'].shift(12).loc[val_start_date:]

# Zapis
output_df = val_data[['CPI', 'Pred_XGBoost', 'Pred_RandomForest', 'Pred_SARIMA', 'Pred_Naive']].reset_index()
output_df.columns = ['date', 'Actual', 'XGBoost', 'RandomForest', 'SARIMA', 'Naive']
output_df.to_csv("validation_data.csv", index=False)

print(">>> Gotowe! Zapisano: validation_data.csv")