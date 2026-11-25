# --- ARCHIWUM (praca inżynierska) – uruchamiaj z głównego folderu: python archiwum/skrypty/<plik>.py
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_sys.path.insert(0, _ROOT)                                   # data_loader z głównego folderu
_os.makedirs(_os.path.join(_ROOT, "archiwum", "wyniki"), exist_ok=True)
_os.chdir(_os.path.join(_ROOT, "archiwum", "wyniki"))        # wykresy/CSV zapisują się w archiwum/wyniki
# ---
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from data_loader import load_data
import matplotlib.pyplot as plt

# 1. Wczytanie danych CPI
df = load_data()          # DataFrame z indeksem = date, kolumna "CPI"
y = df["CPI"]

# 2. Podział na train / test
split_date = "2023-01-01"
y_train = y.loc[:split_date]
y_test = y.loc[split_date:]

print(">>> Zakres TRAIN:")
print("   od:", y_train.index.min().date(), "do:", y_train.index.max().date())
print(">>> Zakres TEST:")
print("   od:", y_test.index.min().date(), "do:", y_test.index.max().date())

# 3. Model naiwny: ostatnia wartość z okresu treningowego
last_value = y_train.iloc[-1]
last_date = y_train.index[-1]

print("\n>>> Ostatnia wartość TRAIN (na której opiera się cały model Naive):")
print(f"   data: {last_date.date()}, CPI: {last_value:.2f}")

# Prognoza: ta sama wartość powtórzona tyle razy, ile jest miesięcy w teście
y_pred = np.repeat(last_value, len(y_test))

# 4. Metryki błędu
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("\n>>> Jakość prognozy Naive na okresie TEST:")
print("   MAE :", mae)
print("   RMSE:", rmse)

# 5. Podgląd: pierwsze 5 miesięcy testowych z prognozą
results_preview = pd.DataFrame({
    "CPI_real": y_test,
    "CPI_naive_pred": y_pred
})
results_preview["abs_error"] = (results_preview["CPI_real"] - results_preview["CPI_naive_pred"]).abs()

print("\n>>> Pierwsze 5 obserwacji TEST (rzeczywiste vs Naive):")
print(results_preview.head())

# 6. Wykres: TRAIN vs TEST vs Naive
plt.figure(figsize=(10, 5))
plt.plot(y_train.index, y_train, label="Training (history)")
plt.plot(y_test.index, y_test, label="Testing (real CPI)")
plt.plot(y_test.index, y_pred, label="Prediction: Naive Forecast")

plt.title("Naive Forecast – CPI (train/test)")
plt.xlabel("Data")
plt.ylabel("CPI (indeks)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.close()
