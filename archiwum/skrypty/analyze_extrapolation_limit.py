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
# Wymuszamy backend nieinteraktywny (Agg), który jest stabilny w IDE
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Lokalizacja plików CSV (wymaga istnienia cpi_data.csv i validation_data.csv)
try:
    hist_df = pd.read_csv(_os.path.join(_ROOT, "cpi_data.csv"), parse_dates=['date']).set_index('date')
    val_df = pd.read_csv(_os.path.join(_ROOT, "validation_data.csv"), parse_dates=['date']).set_index('date')
except FileNotFoundError:
    print("BŁĄD: Upewnij się, że pliki cpi_data.csv i validation_data.csv istnieją.")
    exit()

# === STAŁA ANALITYCZNA ===
# Data zakończenia zbioru TRENINGOWEGO dla modeli walidacyjnych (przed 2015-01-01)
TRAIN_END_DATE = "2015-01-01"
PLOT_START_DATE = "2018-01-01"

# 1. Znajdowanie "Szklanego Sufitu" (Max CPI widziane przez model w treningu)
train_subset = hist_df.loc[:TRAIN_END_DATE]['CPI']
MAX_CPI_TRAINING = train_subset.max()

# 2. Filtracja danych do wykresu
val_df_plot = val_df.loc[PLOT_START_DATE:].copy()
val_df_plot['XGBoost'] = val_df_plot['XGBoost'].replace(0, np.nan)

# 3. Tworzenie wykresu
plt.figure(figsize=(14, 7))

# Wykres 1: Rzeczywistość (Czarna linia)
plt.plot(val_df_plot.index, val_df_plot['Actual'], label="Rzeczywisty CPI (Actual)", color='black', linewidth=3)

# Wykres 2: Predykcja XGBoost (Czerwona linia - pokazuje limit ekstrapolacji)
plt.plot(val_df_plot.index, val_df_plot['XGBoost'], label="Predykcja XGBoost (Brak ekstrapolacji)", color='red', linestyle='--', linewidth=2)

# Wykres 3: Szklany Sufit (Linia Graniczna Treningu)
plt.axhline(y=MAX_CPI_TRAINING, color='#3498db', linestyle='-', label=f"Limit znajomości danych (Max CPI w Training Set)")

# 4. Adnotacje i styl
plt.text(pd.to_datetime("2020-01-01"),
         MAX_CPI_TRAINING + 0.2,
         f"LIMIT WIEDZY MODELI DRZEWIASTYCH (Max CPI={MAX_CPI_TRAINING:.2f})",
         color='#2980b9',
         fontsize=10,
         fontweight='bold')

plt.axvspan(pd.to_datetime("2021-01-01"),
            pd.to_datetime("2024-01-01"),
            facecolor='#e74c3c',
            alpha=0.1,
            label='Okres Szoku/Nieznany')

plt.title("Analiza Ograniczenia Ekstrapolacji: MAE vs. Limit Danych Treningowych")
plt.xlabel("Data")
plt.ylabel("Wskaźnik CPI")
plt.legend(loc='upper left')
plt.grid(alpha=0.4)
plt.tight_layout()

# === KLUCZOWA ZMIANA ===
# Zamiast plt.close() zapisujemy plik, aby ominąć błąd backendu IDE
output_filename = "extrapolation_limit_analysis.png"
plt.savefig(output_filename, dpi=300)
print(f"[INFO] Wykres zapisano do pliku: {output_filename}")
# ========================