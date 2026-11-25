import matplotlib

matplotlib.use("Agg")
import os
os.makedirs("wykresy", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data

history = load_data()

try:
    # SARIMA
    sarima_df = pd.read_csv("sarima_10y_forecast.csv")
    sarima_df['date'] = pd.to_datetime(sarima_df['date'])
    sarima_df = sarima_df.set_index('date')

    # ML (XGBoost + Random Forest)
    ml_df = pd.read_csv("ml_10y_forecast_combined.csv")
    ml_df['date'] = pd.to_datetime(ml_df['date'])
    ml_df = ml_df.set_index('date')

    # Naive
    naive_df = pd.read_csv("naive_10y_forecast.csv")
    naive_df['date'] = pd.to_datetime(naive_df['date'])
    naive_df = naive_df.set_index('date')

    print(">>> Wczytano wszystkie prognozy.")

except FileNotFoundError as e:
    print(f"!!! Brakuje pliku: {e}")
    exit()

plt.figure(figsize=(15, 9))

# Historia
start_plot_date = "2017-01-01"
history_subset = history.loc[start_plot_date:]
plt.plot(history_subset.index, history_subset['CPI'], label="Historia (NBP)", color='black', linewidth=3)

# Naive
plt.plot(naive_df.index, naive_df['Naive_Simple'], label="Naive Simple", color='green', linestyle='--', alpha=0.6)
plt.plot(naive_df.index, naive_df['Naive_Seasonal'], label="Naive Seasonal", color='purple', linestyle='-.', alpha=0.6)

# SARIMA
plt.plot(sarima_df.index, sarima_df['CPI_forecast'], label="SARIMA", color='blue', linewidth=2)
plt.fill_between(sarima_df.index, sarima_df['lower_95'], sarima_df['upper_95'], color='blue', alpha=0.05)

# ML: XGBoost
plt.plot(ml_df.index, ml_df['CPI_forecast_XGB'], label="XGBoost", color='red', linewidth=2.5)

# ML: Random Forest (NOWOŚĆ - Ciemnożółty/Złoty)
plt.plot(ml_df.index, ml_df['CPI_forecast_RF'], label="Random Forest", color='#DAA520', linewidth=2.5, linestyle='-')

# Linie pomocnicze
plt.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=102.5, color='orange', linestyle=':', alpha=0.8, label="Cel inflacyjny NBP")

plt.title("ZESTAWIENIE FINALNE + Random Forest", fontsize=16)
plt.xlabel("Rok")
plt.ylabel("CPI")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("wykresy/FINAL_COMPARISON_WITH_RF.png", dpi=300)
print(">>> Zapisano: FINAL_COMPARISON_WITH_RF.png")
plt.close()