import subprocess
import sys
import time
import os

# Lista skryptów do uruchomienia w kolejności (Rurociąg danych)
scripts_to_run = [
    "data_source.py",  # 1. Pobierz najnowsze dane z NBP
    "arima_auto.py",  # 2. Przelicz model SARIMA (AutoARIMA)
    "ml_forecast_10y.py",  # 3. Przelicz modele ML (XGBoost, RF) na pełnych danych
    "naive_forecast_10y.py",  # 4. Wygeneruj benchmarki (Naive)
    "generate_validation.py",  # 5. Przelicz walidację historyczną (do zakładki Ewaluacja)
    "compare_forecasts.py",  # 6. Wygeneruj statyczny wykres PNG do pracy
    "experiments.py",  # 7. [PROJEKT AI] Eksperymenty E1-E6: baseline + warianty LSTM (~10-20 min)
    "xai_lstm.py",  # 8. [PROJEKT AI] Warstwa XAI + analiza błędów
    "lstm_forecast_future.py",  # 9. [PROJEKT AI] Finalny model LSTM i prognoza 12 mies.
    "final_evaluation.py",  # 10. [PROJEKT AI] Ocena końcowa na danych 11.2025-10.2026
    "report_figures.py",  # 11. [PROJEKT AI] Wykresy do sprawozdań (KM2 + KM3)
]


def run_script(script_name):
    """Uruchamia pojedynczy skrypt i mierzy czas."""
    print(f"\n{'=' * 60}")
    print(f"🚀 URUCHAMIAM: {script_name}...")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Sprawdź czy plik istnieje
    if not os.path.exists(script_name):
        print(f"❌ BŁĄD: Nie znaleziono pliku '{script_name}'!")
        print("   Upewnij się, że jesteś w dobrym folderze.")
        sys.exit(1)

    try:
        # Uruchomienie skryptu w tym samym środowisku Python
        result = subprocess.run([sys.executable, script_name], check=True)

        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ SUKCES: {script_name} zakończony w {duration:.2f} sek.")

    except subprocess.CalledProcessError:
        print(f"\n❌ BŁĄD KRYTYCZNY w pliku: {script_name}")
        print("   Zatrzymuję cały proces. Popraw błąd i spróbuj ponownie.")
        sys.exit(1)


def main():
    total_start = time.time()
    print("\n>>> ROZPOCZYNAM PEŁNĄ AKTUALIZACJĘ SYSTEMU (PIPELINE) <<<")
    print("To może chwilę potrwać (szczególnie AutoARIMA i Walidacja)...\n")

    for script in scripts_to_run:
        run_script(script)

    total_end = time.time()
    total_duration = total_end - total_start

    print(f"\n{'#' * 60}")
    print(f"🎉 WSZYSTKO GOTOWE!")
    print(f"   Cały proces zajął: {total_duration:.2f} sek ({total_duration / 60:.1f} min)")
    print(f"   Dane są zaktualizowane. Możesz uruchomić: streamlit run app.py")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()