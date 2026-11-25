import pandas as pd
from pathlib import Path

# Nowy plik, który wygeneruje data_source.py
CSV_PATH = Path(__file__).parent / 'cpi_data.csv'


def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku {CSV_PATH}. Uruchom najpierw data_source.py!")

    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.set_index('date').sort_index()
    return df


if __name__ == '__main__':
    # Check
    try:
        cpi = load_data()
        print(">>> Test loadera udany.")
        print(f"Zakres: {cpi.index.min().date()} -> {cpi.index.max().date()}")
        print(f"Liczba wierszy: {len(cpi)}")
    except Exception as e:
        print("Błąd:", e)