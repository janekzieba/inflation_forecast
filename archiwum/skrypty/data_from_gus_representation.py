import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import modułu do obsługi dat
from pathlib import Path

CSV_PATH = Path("cpi_data.csv")

# -----------------------------------------
# 1. Wczytanie danych
# -----------------------------------------
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.set_index("date").sort_index()

# -----------------------------------------
# 2. Rysowanie wykresu
# -----------------------------------------
fig, ax = plt.subplots(figsize=(14,5))

ax.plot(df.index, df["CPI"], linewidth=2, label="CPI (wartości źródłowe)")

# pasmo celu inflacyjnego
ax.axhspan(101.5, 103.5, color="green", alpha=0.15,
           label="Cel inflacyjny NBP (2.5% ±1 p.p.)")

# szoki makro
covid  = pd.to_datetime("2020-03-01")
war    = pd.to_datetime("2022-02-24")

ax.axvline(covid, color="red",    linestyle="--", linewidth=2, label="COVID-19 (2020)")
ax.axvline(war,   color="purple", linestyle="--", linewidth=2, label="Wojna w Ukrainie (2022)")

# --- NOWE: Ustawienie oznaczenia każdego roku na osi X ---
ax.xaxis.set_major_locator(mdates.YearLocator())    # Tick w każdym roku
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) # Formatowanie etykiety jako sam rok

# Odcinamy wykres dokładnie do ostatniej dostępnej daty
ax.set_xlim(df.index.min(), df.index.max())

# opisy
ax.set_ylabel("CPI")
ax.set_xlabel("Rok")
ax.set_title("Poziom CPI w Polsce na tle celu inflacyjnego NBP oraz szoków makroekonomicznych")

# legenda
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper left")

fig.tight_layout()
fig.savefig("cpi_final_plot.png", dpi=300, bbox_inches="tight")

print("📁 Zapisano wykres: cpi_final_plot.png")