import matplotlib
matplotlib.use("Agg")  # bez okien – pipeline nie blokuje się
import os
os.makedirs("wykresy", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from data_loader import load_data

# Wczytanie danych
df = load_data()
y = df["CPI"]

print(f">>> Wczytano dane: {len(y)} obserwacji.")

# 2. AUTOMATYCZNY DOBÓR MODELU (AutoARIMA) - auto sprawdzenie czy dane trzeba roznicowac, dobranie p i q
print(">>> Rozpoczynam wyszukiwanie najlepszego modelu auto_arima...")
print("    (To może chwilę potrwać, bądź cierpliwy)...")

model = pm.auto_arima(
    y,
    start_p=0, start_q=0,
    max_p=3, max_q=3,
    m=12,              # Sezonowość roczna (12 miesięcy)
    start_P=0, seasonal=True,
    d=None,            # Auto ustaalenie rzedu roznicowania test dickeya-fullera
    D=1,               # Wymuszamy jedno różnicowanie sezonowe (przez inflacje)
    trace=True,        # Wypisanie w konsoli dzialania
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True      # Algorytm stepwise (szybszy i mądrzejszy niż grid search)
)

print("\n>>> ZNALEZIONO NAJLEPSZY MODEL:")
print(model.summary())

# 3. DIAGNOSTYKA (Wykresy reszt) - okno z 4 wykreaami
# Pokazanie, czy błędy modelu są losowe (szum biały).
print(">>> Wyświetlam wykresy diagnostyczne...")
model.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.savefig("wykresy/sarima_diagnostyka.png", dpi=150)
plt.close()

# 4. PROGNOZA NA 10 LAT (120 miesięcy)
horizon = 120
print(f"\n>>> Generowanie prognozy na {horizon} miesięcy...")

forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)

# Tworzenie indeksu dat dla przyszłości
last_date = y.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

# Zapis wyników do DataFrame (żeby pasowało do compare_forecasts.py)
forecast_df = pd.DataFrame({
    "date": future_dates,
    "CPI_forecast": forecast,
    "lower_95": conf_int[:, 0],
    "upper_95": conf_int[:, 1]
})

# 5. ZAPIS DO PLIKU
csv_filename = "sarima_10y_forecast.csv"
forecast_df.to_csv(csv_filename, index=False)
print(f">>> Zapisano prognozę do pliku: {csv_filename}")

# 6. SZYBKI PODGLĄD WYKRESU
plt.figure(figsize=(12, 6))
plt.plot(y.index, y, label="Historia")
plt.plot(forecast_df["date"], forecast_df["CPI_forecast"], label="Prognoza AutoARIMA", color='blue')
plt.fill_between(forecast_df["date"],
                 forecast_df["lower_95"],
                 forecast_df["upper_95"],
                 color='blue', alpha=0.1)
plt.title(f"AutoARIMA Forecast: {model.order}x{model.seasonal_order}")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("wykresy/sarima_10y_forecast.png", dpi=150)
plt.close()
print(">>> Zapisano wykresy w folderze wykresy/")