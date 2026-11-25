import matplotlib
matplotlib.use("Agg")
import os
os.makedirs("wykresy", exist_ok=True)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data

# 1. Wczytanie pełnych danych
df = load_data()
y = df["CPI"]
last_date = y.index[-1]

print(f">>> Ostatnia data w danych: {last_date.date()}")
print(f">>> Ostatnia wartość CPI:   {y.iloc[-1]}")

# 2. Ustawienia horyzontu (10 lat = 120 miesięcy)
horizon = 120
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=horizon,
    freq='MS'
)

# ==========================================
# A. PROGNOZA 1: SIMPLE NAIVE (Płaska linia)
# ==========================================
# ostatnią wartość * 120
last_value = y.iloc[-1]
forecast_simple = np.repeat(last_value, horizon)

# ==========================================
# B. PROGNOZA 2: SEASONAL NAIVE (Kopiowanie roku)
# ==========================================
# Bierzemy ostatnie 12 miesięcy historii
last_12_months_values = y.iloc[-12:].values

# Powtarzamy ten cykl 10x ( 10 lat * 12 miesięcy = 120)
forecast_seasonal = np.tile(last_12_months_values, int(horizon / 12))

# ==========================================
# 3. ZAPIS DO DATAFRAME I PLIKU
# ==========================================
naive_df = pd.DataFrame({
    "date": future_dates,
    "Naive_Simple": forecast_simple,
    "Naive_Seasonal": forecast_seasonal
})


naive_df.to_csv("naive_10y_forecast.csv", index=False)
print(">>> Zapisano prognozy naiwne do: naive_10y_forecast.csv")

# ==========================================
# 4. WYKRES
# ==========================================
plt.figure(figsize=(12, 6))

# Historia (od 2020 roku dla czytelności)
history_subset = y.loc["2020-01-01":]
plt.plot(history_subset.index, history_subset, label="Historia (2020-2025)", color='black')

# Prognozy
plt.plot(naive_df["date"], naive_df["Naive_Simple"],
         label="Simple Naive (Stała)", color='green', linestyle='--', linewidth=2)

plt.plot(naive_df["date"], naive_df["Naive_Seasonal"],
         label="Seasonal Naive (Powielony rok)", color='purple', linestyle='-.', alpha=0.7)

plt.title("Prognoza Naiwna (Benchmark) na 10 lat")
plt.xlabel("Rok")
plt.ylabel("CPI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("wykresy/naive_10y_forecast.png")
print(">>> Zapisano wykres do: wykresy/naive_10y_forecast.png")
plt.close()