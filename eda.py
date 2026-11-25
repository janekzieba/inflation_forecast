"""
Analiza eksploracyjna szeregu CPI (praca inżynierska, rozdz. EDA) – połączone dawne EDA_1.py, eda.py i eda_2.py.
  1. dekompozycja szeregu (trend / sezonowość / reszta)  -> wykresy/eda_decomposition.png
  2. ACF i PACF po różnicowaniu sezonowym                -> wykresy/eda_acf_pacf.png
  3. testy stacjonarności ADF i KPSS                     -> wykresy/stacjonarnosc_adf_kpss.csv
Uruchomienie: python eda.py
"""
import os
os.makedirs("wykresy", exist_ok=True)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from data_loader import load_data

# 1. Wczytanie danych (miesięcznych)
df = load_data()
series = df['CPI'].dropna()
print(f">>> Generowanie EDA dla {len(series)} obserwacji...")

# ==========================================
# 1. DEKOMPOZYCJA (Trend i Sezonowość)
# ==========================================
# period=12 -> sezonowość roczna przy danych miesięcznych
result = seasonal_decompose(series, model='additive', period=12)

fig = result.plot()
fig.set_size_inches(12, 10)

# Opis osi: oś X to czas (miesiące, pokazane jako daty)
# Zwykle wystarczy opisać oś X tylko na dole.
fig.axes[-1].set_xlabel("Rok")  # lub: "Czas (miesiące)"
# Opcjonalnie: doprecyzuj oś Y dla pierwszego panelu:
fig.axes[0].set_ylabel("CPI [pkt indeksu]")

plt.tight_layout()
plt.savefig("wykresy/eda_decomposition.png", dpi=200)
print(">>> Zapisano: wykresy/eda_decomposition.png")
plt.close()

# ==========================================
# 2. ACF / PACF (Autokorelacja)
# ==========================================
# Różnicowanie sezonowe (12): usunięcie sezonowości rocznej
series_diff = series.diff(12).dropna()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series_diff, lags=40, ax=ax1, title="Autokorelacja (ACF) - różnicowanie sezonowe (D=1)")
plot_pacf(series_diff, lags=40, ax=ax2, title="Autokorelacja cząstkowa (PACF) - różnicowanie sezonowe (D=1)")

ax1.set_xlabel("Opóźnienie [miesiące]")
ax2.set_xlabel("Opóźnienie [miesiące]")
ax1.set_ylabel("ACF")
ax2.set_ylabel("PACF")

plt.tight_layout()
plt.savefig("wykresy/eda_acf_pacf.png", dpi=200)
print(">>> Zapisano: eda_acf_pacf.png")
plt.close()


# ==========================================
# 3. TESTY STACJONARNOŚCI (ADF, KPSS) – dawne eda_2.py
# ==========================================
from statsmodels.tsa.stattools import adfuller, kpss

y = df["CPI"].astype(float)

def adf_p(x):
    return adfuller(x.dropna(), autolag="AIC")[1]

def kpss_p(x):
    # H0: stacjonarność (poziom), więc mały p-value = odrzucamy stacjonarność
    return kpss(x.dropna(), regression="c", nlags="auto")[1]

tests = {
    r"$y_t$": y,
    r"$\Delta y_t$ (diff(1))": y.diff(),
    r"$\Delta_{12}y_t$ (diff(12))": y.diff(12),
    r"$\Delta\Delta_{12}y_t$ (diff(12).diff(1))": y.diff(12).diff(),
}

rows = []
for name, s in tests.items():
    rows.append({
        "Szereg": name,
        "ADF_pvalue": adf_p(s),
        "KPSS_pvalue": kpss_p(s),
    })

out = pd.DataFrame(rows)
print(out)

out.to_csv("wykresy/stacjonarnosc_adf_kpss.csv", index=False)
print("Zapisano: wykresy/stacjonarnosc_adf_kpss.csv")
