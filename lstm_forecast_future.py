"""
Finalny model LSTM do aplikacji: trening na wszystkich dostępnych danych (ostatnie 24 mies. = walidacja
dla early stopping), ensemble z 5 ziaren, prognoza 12 miesięcy naprzód z przedziałem MC-dropout
oraz wyjaśnienie (Integrated Gradients) bieżącej prognozy.

Uruchomienie (po experiments.py, które wybiera wariant i hiperparametry):
    python lstm_forecast_future.py
Wynik: lstm_forecast.csv, lstm_forecast_explanation.csv (czyta je app.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_utils import load_series
from lstm_model import LSTMForecaster, make_config
from xai_lstm import integrated_gradients
from experiments import validation_errors

H = 12
SEEDS = range(5)
RES = Path("results")


def main():
    y = load_series()
    if (RES / "best_lstm.json").exists():
        variant = json.loads((RES / "best_lstm.json").read_text())["B"]
        base = json.loads((RES / "best_config.json").read_text())["B"]
    else:
        print("!!! Brak results/ – używam ustawień domyślnych (uruchom najpierw experiments.py)")
        variant, base = "LSTM-Δ", dict(window=12, hidden=32, dropout=0.2, lr=1e-3)
    print(f">>> Finalny model: {variant}, {base}")

    origin = np.array([len(y) - 1])
    points, samples, attrs, qs = [], [], [], []
    for seed in SEEDS:
        f = LSTMForecaster(make_config(variant, base, seed=seed, H=H)).fit(y, len(y))
        out = f.forecast(y, origin, H=H, mc_samples=200)
        points.append(out["point"][0])
        samples.append(out["samples"][0])
        a, _ = integrated_gradients(f, f.input_tensor(y, origin))
        attrs.append(a[0, :, 0])
        errs = validation_errors(f, y, len(y))
        qs.append([np.quantile(errs[h], 0.95) for h in range(1, H + 1)])
        f.save(RES / "models" / f"FINAL_{variant}_seed{seed}.pt")

    S = np.concatenate(samples, axis=0)         # (seeds*mc, H)
    dates = pd.date_range(y.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")
    fc = pd.DataFrame({"date": dates, "LSTM": np.mean(points, axis=0),
                       "LSTM_lo": np.percentile(S, 2.5, axis=0), "LSTM_hi": np.percentile(S, 97.5, axis=0),
                       "LSTM_seed_std": np.std(points, axis=0)})
    # Kalibracja przedziału: błędy out-of-sample tego samego wariantu z testu Podziału B (2023-…),
    # czyli najświeższy okres, którego modele z eksperymentu nie widziały. Fallback: walidacja finalnego modelu.
    q = np.mean(qs, axis=0)
    calib = "walidacja finalnego modelu (ostatnie 24 mies.)"
    if (RES / "forecasts.csv").exists():
        fcx = pd.read_csv(RES / "forecasts.csv")
        g = fcx[(fcx.split == "B") & (fcx.model == variant)]
        g = g.groupby(["origin", "h"], as_index=False)[["y_true", "y_pred"]].mean()
        q = np.array([np.quantile(np.abs(g[g.h == h].y_true - g[g.h == h].y_pred), 0.95) for h in range(1, H + 1)])
        calib = "błędy out-of-sample z testu Podziału B (2023 – koniec danych)"
    fc["LSTM_lo_conf"], fc["LSTM_hi_conf"] = fc["LSTM"] - q, fc["LSTM"] + q
    fc.to_csv("lstm_forecast.csv", index=False)

    A = np.mean(attrs, axis=0)
    w = len(A)
    expl = pd.DataFrame({"input_month": y.index[-w:], "lag": np.arange(w, 0, -1), "CPI": y.values[-w:],
                         "delta": y.diff().values[-w:], "contribution_pp": A})
    expl.to_csv("lstm_forecast_explanation.csv", index=False)
    Path("lstm_forecast_meta.json").write_text(json.dumps({"variant": variant, "config": base,
                                                           "data_end": str(y.index[-1].date()),
                                                           "interval_calibration": calib},
                                                          ensure_ascii=False))
    print(fc.to_string(index=False, float_format="%.2f"))
    print(">>> Zapisano: lstm_forecast.csv, lstm_forecast_explanation.csv")


if __name__ == "__main__":
    main()
