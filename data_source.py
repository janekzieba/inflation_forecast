import requests
import pandas as pd
from io import BytesIO

URL = "https://static.nbp.pl/dane/inflacja/bazowa.xlsx"
LOCAL_FILE = "bazowa-2.xlsx"
OUT_CSV = "cpi_data.csv"


def parse_polish_date(date_val):

    # Reczny parser dla polskich dat sty 01, niezaleznie od systemu komp. jak juz zmienione na date timestamp, to return to samo
    if isinstance(date_val, pd.Timestamp):
        return date_val

    text = str(date_val).lower().strip()

    # Mapa polskich skrótów na numery miesięcy
    pl_months = {
        'sty': '01', 'lut': '02', 'mar': '03', 'kwi': '04', 'maj': '05', 'cze': '06',
        'lip': '07', 'sie': '08', 'wrz': '09', 'paź': '10', 'lis': '11', 'gru': '12',
        # Na wypadek gdyby Excel zgubił polskie znaki:
        'paz': '10'
    }

    try:
        # Dzielimy "sty 01"
        parts = text.split()
        if len(parts) != 2:
            return pd.NaT  # Błąd, to nie data

        month_str = parts[0]
        year_str = parts[1]

        # Sprawdzamy czy miesiąc jest w słowniku
        if month_str not in pl_months:
            return pd.NaT

        month_num = pl_months[month_str]

        # Zamiana roku '01' na '2001', '99' na '1999'
        year_int = int(year_str)
        full_year = 2000 + year_int if year_int < 100 else year_int

        return pd.to_datetime(f"{full_year}-{month_num}-01")

    except Exception:
        return pd.NaT


def process_df(df):
    # Bierzemy kolumny A (0) i B (1)
    sub = df.iloc[3:, [0, 1]].copy()

    sub.columns = ["date_raw", "CPI_raw"]

    # Usuwamy puste wiersze
    sub = sub.dropna(subset=["date_raw"])

    # Używamy naszego "Tłumacza" do dat
    sub["date"] = sub["date_raw"].apply(parse_polish_date)

    # Usuwamy wiersze, gdzie data się nie udała
    sub = sub.dropna(subset=["date"])

    # Konwersja CPI (zamiana przecinka na kropkę)
    sub["CPI"] = (
        sub["CPI_raw"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    return sub[["date", "CPI"]].reset_index(drop=True)


def load_manual():
    print(">>> Używam lokalnego pliku:", LOCAL_FILE)
    # header=None, bo sami tniemy indeksami (iloc)
    df = pd.read_excel(LOCAL_FILE, sheet_name="dane_miesieczne", header=None)
    return process_df(df)


def load_automatic():
    print(">>> Pobieram plik z NBP...")
    response = requests.get(URL, timeout=10)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content), sheet_name="dane_miesieczne", header=None)
    return process_df(df)


def main():
    try:
        cpi_monthly = load_automatic()
        print(">>> Automatyczne pobieranie udane!")
    except Exception as e:
        print("!!! Błąd automatycznego pobierania:", e)
        print(">>> Próba odczytu manualnego...")
        try:
            cpi_monthly = load_manual()
        except Exception as e_loc:
            print(f"!!! Błąd pliku lokalnego: {e_loc}")
            return

    print(f"\nZakres pobranych danych (Musi być od 2001-01-01!):")
    if not cpi_monthly.empty:
        print(f"Od: {cpi_monthly['date'].min().date()}")
        print(f"Do: {cpi_monthly['date'].max().date()}")
        print(f"Liczba obserwacji: {len(cpi_monthly)}")
    else:
        print("!!! UWAGA: Pusty DataFrame. Coś poszło nie tak z formatem.")

    cpi_monthly.to_csv(OUT_CSV, index=False)
    print(f"\n>>> Zapisano dane do: {OUT_CSV}")


main()