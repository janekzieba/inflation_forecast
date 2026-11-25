# --- ARCHIWUM (praca inżynierska) – uruchamiaj z głównego folderu: python archiwum/skrypty/<plik>.py
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_sys.path.insert(0, _ROOT)                                   # data_loader z głównego folderu
_os.makedirs(_os.path.join(_ROOT, "archiwum", "wyniki"), exist_ok=True)
_os.chdir(_os.path.join(_ROOT, "archiwum", "wyniki"))        # wykresy/CSV zapisują się w archiwum/wyniki
# ---
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Importy modeli
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR  # Support Vector Regression
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Importy do inżynierii danych
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Wymuszamy backend (żeby nie było błędu w PyCharm)
matplotlib.use('Agg')

from data_loader import load_data

# =========================================================
# 1. PRZYGOTOWANIE DANYCH
# =========================================================
def create_features(df):
    df = df.copy()
    df['lag_1'] = df['CPI'].shift(1)
    df['lag_2'] = df['CPI'].shift(2)
    df['lag_12'] = df['CPI'].shift(12)
    df['roll_mean_3'] = df['CPI'].shift(1).rolling(window=3).mean()
    df['month'] = df.index.month
    df = df.dropna()
    return df

df_raw = load_data()
df_features = create_features(df_raw)

features = ['lag_1', 'lag_2', 'lag_12', 'roll_mean_3', 'month']
target = 'CPI'

X = df_features[features]
y = df_features[target]

# Podział taki sam jak w ewaluacji (2015)
split_date = "2015-01-01"

X_train = X.loc[:split_date]
y_train = y.loc[:split_date]
X_test = X.loc[split_date:]
y_test = y.loc[split_date:]

print(f"--- TRENING WIELU MODELI ---")
print(f"Train: do {split_date}")
print(f"Test: {len(X_test)} miesięcy")

# Kontener na wyniki
preds = {}
preds['Actual'] = y_test

# =========================================================
# 2. TRENING MODELI
# =========================================================

# --- A. MODELE DRZEWIASTE (Bez skalowania) ---
print("1. Trenowanie XGBoost...")
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
preds['XGBoost'] = xgb.predict(X_test)

print("2. Trenowanie Random Forest...")
rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds['RandomForest'] = rf.predict(X_test)

# --- B. MODELE WYMAGAJĄCE SKALOWANIA (SVR, K-NN, LR) ---
# Używamy Pipeline, żeby automatycznie przeskalować dane przed wejściem do modelu
print("3. Trenowanie SVR (z automatycznym skalowaniem)...")
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
svr.fit(X_train, y_train)
preds['SVR'] = svr.predict(X_test)

print("4. Trenowanie K-NN (z automatycznym skalowaniem)...")
knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
knn.fit(X_train, y_train)
preds['K-NN'] = knn.predict(X_test)

print("5. Trenowanie Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
preds['LinearReg'] = lr.predict(X_test)

# --- C. MODELE SPECJALNE (SARIMA, NAIVE) ---
print("6. Obliczanie SARIMA...")
# Uwaga: SARIMAX z statsmodels wymaga odpowiedniego podejścia do dat
sarima = SARIMAX(df_raw['CPI'], order=(2, 0, 1), seasonal_order=(2, 1, 0, 12), enforce_stationarity=False).fit(disp=False)
# Predykcja na okres testowy
preds['SARIMA'] = sarima.predict(start=split_date, end=df_raw.index[-1]).loc[X_test.index]

print("7. Obliczanie Naive...")
preds['Naive'] = df_raw['CPI'].shift(12).loc[X_test.index]


# =========================================================
# 3. ZAPIS DO PLIKU I OBLICZENIE METRYK (Z RMSE!)
# =========================================================
results_df = pd.DataFrame(preds, index=X_test.index)
results_df.index.name = 'date'
results_df.reset_index(inplace=True)

# Zapisujemy wszystkie wyniki do pliku, który wczytuje aplikacja
results_df.to_csv("validation_data_ML_all.csv", index=False)
print(">>> Zapisano wyniki wszystkich modeli do: validation_data_ML_all.csv")

# Obliczanie błędów do wyświetlenia teraz
metrics = []
for m in ['XGBoost', 'RandomForest', 'SVR', 'K-NN', 'LinearReg', 'SARIMA', 'Naive']:
    mae = mean_absolute_error(preds['Actual'], preds[m])
    rmse = sqrt(mean_squared_error(preds['Actual'], preds[m]))
    metrics.append({'Model': m, 'MAE': mae, 'RMSE': rmse})

metrics_df = pd.DataFrame(metrics).sort_values('MAE')

print("\n=== RANKING BŁĘDÓW (2015-2025) ===")
print(metrics_df.round(4))

# =========================================================
# 4. GENEROWANIE WYKRESÓW DOWODOWYCH
# =========================================================

# --- WYKRES 1: MAE ---
plt.figure(figsize=(12, 6))
colors_mae = ['#2ecc71' if x < 1.0 else '#e74c3c' if x > 3.0 else '#3498db' for x in metrics_df['MAE']]
bars_mae = plt.bar(metrics_df['Model'], metrics_df['MAE'], color=colors_mae)

plt.title('Porównanie Błędów MAE')
plt.ylabel('MAE (Mean Absolute Error)')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Próg akceptowalności')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Dodanie wartości na słupkach
for bar in bars_mae:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("full_model_comparison_mae.png", dpi=300)
print("[INFO] Wykres MAE zapisano jako: full_model_comparison_mae.png")

# --- WYKRES 2: RMSE (DODANY!) ---
# Sortujemy df po RMSE dla czytelności drugiego wykresu
metrics_df_rmse = metrics_df.sort_values('RMSE')

plt.figure(figsize=(12, 6))
# Kolorystyka: zielony (niski błąd) -> czerwony (wysoki błąd)
colors_rmse = ['#2ecc71' if x < 1.5 else '#e74c3c' if x > 5.0 else '#9b59b6' for x in metrics_df_rmse['RMSE']]
bars_rmse = plt.bar(metrics_df_rmse['Model'], metrics_df_rmse['RMSE'], color=colors_rmse)

plt.title('Porównanie Błędów RMSE')
plt.ylabel('RMSE (Root Mean Squared Error)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Dodanie wartości na słupkach
for bar in bars_rmse:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("full_model_comparison_rmse.png", dpi=300)
print("[INFO] Wykres RMSE zapisano jako: full_model_comparison_rmse.png")