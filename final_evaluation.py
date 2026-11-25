"""
Kamień milowy 3: końcowa ocena na danych NIGDY nieużywanych w projekcie (11.2025 – 10.2026).

Wszystkie modele są uczone wyłącznie na danych do 10.2025 (dokładnie te, na których powstał model finalny
w kamieniu milowym 2) i ZAMROŻONE. Następnie:
  1. ocena ex ante – prognoza na 12 miesięcy wykonana w 10.2025 (lstm_forecast.csv) vs rzeczywistość,
  2. ocena krocząca (rolling origin) – co miesiąc nowa obserwacja trafia na wejście, parametry bez zmian,
  3. XAI na nowych danych – Integrated Gradients dla każdej prognozy z okresu 11.2025 – 10.2026.

Dane holdout: holdout_cpi_2025-11_2026-10.csv (odczyty finalne GUS, CPI r/r, analogiczny miesiąc = 100).
Uruchomienie (po experiments.py i lstm_forecast_future.py):
    python final_evaluation.py
Wyniki -> results/final/
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_utils import (load_series, to_long, compute_metrics, dm_test, build_train_matrix, recursive_forecast,
                        make_ml_models)
from lstm_model import LSTMForecaster, make_config
from xai_lstm import integrated_gradients

warnings.filterwarnings("ignore")
RES = Path("results")
OUT = RES / "final"
OUT.mkdir(parents=True, exist_ok=True)
H = 12
HS = [1, 3, 6]
SEEDS = range(5)


def main():
    y = load_series()                                   # dane uczące: 01.2001 – 10.2025
    ho = pd.read_csv("holdout_cpi_2025-11_2026-10.csv", parse_dates=["date"]).set_index("date")["CPI"]
    ye = pd.concat([y, ho]).asfreq("MS")                # szereg rozszerzony o 12 nowych miesięcy
    T = len(y)
    origins = np.arange(T - 1, len(ye) - 1)             # 10.2025 … 09.2026
    print(f"Trening: {y.index[0].date()} – {y.index[-1].date()} ({T} obs.), holdout: {len(ho)} mies.")

    variant = json.loads((RES / "best_lstm.json").read_text())["B"]
    base = json.loads((RES / "best_config.json").read_text())["B"]
    fcx = pd.read_csv(RES / "forecasts.csv")

    def conf_q(model):
        """Kwantyl 95% |błędów| z testu Podziału B – ta sama kalibracja co w aplikacji."""
        g = fcx[(fcx.split == "B") & (fcx.model == model)].groupby(["origin", "h"], as_index=False)[
            ["y_true", "y_pred"]].mean()
        return np.array([np.quantile(np.abs(g[g.h == h].y_true - g[g.h == h].y_pred), 0.95) for h in range(1, H + 1)])

    parts = []
    # --- modele bazowe ---
    parts.append(to_long("Naive", 0, ye, origins, np.repeat(ye.values[origins][:, None], H, axis=1)))
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(y, order=(2, 0, 1), seasonal_order=(2, 1, 0, 12), enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    P, L, U = [], [], []
    for i in origins:
        f = res.apply(ye.iloc[:i + 1]).get_forecast(H)
        ci = f.conf_int(alpha=0.05).values
        P.append(f.predicted_mean.values), L.append(ci[:, 0]), U.append(ci[:, 1])
    parts.append(to_long("SARIMA", 0, ye, origins, np.array(P), np.array(L), np.array(U)))
    for name in ["LR", "XGBoost"]:
        for target in ["level", "delta"]:
            s, se = (y, ye) if target == "level" else (y.diff(), ye.diff())
            X, Yt = build_train_matrix(s, T)
            m = make_ml_models()[name].fit(X, Yt)
            p = recursive_forecast(m, se, origins, H)
            if target == "delta":
                p = ye.values[origins][:, None] + np.cumsum(p, axis=1)
            parts.append(to_long(name + ("-Δ" if target == "delta" else ""), 0, ye, origins, p))
    print("  modele bazowe gotowe")

    # --- LSTM: model finalny (zapisany w KM2) + warianty porównawcze uczone na tych samych danych ---
    lstm_models = {}
    for v in sorted({variant, "LSTM-L", "LSTM-MO"}):
        fs = []
        for seed in SEEDS:
            path = RES / "models" / f"FINAL_{v}_seed{seed}.pt"
            if path.exists():
                f = LSTMForecaster.load(path)
            else:
                f = LSTMForecaster(make_config(v, base, seed=seed)).fit(y, T)
                f.save(path)
            fs.append(f)
            out = f.forecast(ye, origins, mc_samples=100)
            q = conf_q(v)
            part = to_long(v, seed, ye, origins, out["point"], out["lo"], out["hi"])
            cl = to_long(v, seed, ye, origins, out["point"] - q, out["point"] - q, out["point"] + q)
            part["lo_conf"], part["hi_conf"] = cl["lo"].values, cl["hi"].values
            parts.append(part)
        lstm_models[v] = fs
        print(f"  {v} gotowy")

    fc = pd.concat(parts, ignore_index=True)
    fc.to_csv(OUT / "holdout_forecasts.csv", index=False)
    avg = fc.groupby(["model", "origin", "h", "target_date"], as_index=False)[
        ["y_origin", "y_true", "y_pred", "lo", "hi", "lo_conf", "hi_conf"]].mean()

    # --- metryki ---
    rows = []
    for (m, h), g in avg.groupby(["model", "h"]):
        if h in HS + [12] and len(g) >= 1:
            r = compute_metrics(g.assign(seed=0), y)
            rows.append(dict(model=m, h=h, **r))
    met = pd.DataFrame(rows)
    naive = met[met.model == "Naive"].set_index("h")["MAE"]
    met["RelMAE"] = met.apply(lambda r: r.MAE / naive[r.h], axis=1)
    # stabilność między ziarnami (std MAE)
    sd = fc[fc.model.str.startswith("LSTM")].assign(ae=lambda d: (d.y_true - d.y_pred).abs()) \
        .groupby(["model", "h", "seed"]).ae.mean().groupby(["model", "h"]).std().rename("MAE_std_seeds")
    met = met.merge(sd.reset_index(), on=["model", "h"], how="left")
    met.to_csv(OUT / "holdout_metrics.csv", index=False)

    rows = []
    for m in avg.model.unique():
        for ref in ["Naive", "XGBoost", "SARIMA", "LR-Δ"]:
            if m == ref:
                continue
            for h in [1, 3]:
                a = avg[(avg.model == m) & (avg.h == h)].set_index("target_date")
                b = avg[(avg.model == ref) & (avg.h == h)].set_index("target_date")
                e1, e2 = (a.y_true - a.y_pred).values, (b.y_true - b.y_pred).values
                st, p2, p1 = dm_test(e1, e2, h)
                rows.append(dict(model=m, reference=ref, h=h, n=len(e1), MAE_model=np.abs(e1).mean(),
                                 MAE_ref=np.abs(e2).mean(), DM=st, p_two_sided=p2, p_model_better=p1))
    pd.DataFrame(rows).to_csv(OUT / "holdout_dm.csv", index=False)

    # --- 1. ocena ex ante: prognoza z 10.2025 na 12 mies. ---
    ex = avg[avg.origin == y.index[-1]].pivot_table(index="target_date", columns="model", values="y_pred")
    ex.insert(0, "rzeczywiste", ho.values)
    dep = pd.read_csv("lstm_forecast.csv", parse_dates=["date"]).set_index("date")
    ex["LSTM (aplikacja)"] = dep["LSTM"].values
    ex["LSTM_lo_conf"], ex["LSTM_hi_conf"] = dep["LSTM_lo_conf"].values, dep["LSTM_hi_conf"].values
    ex.to_csv(OUT / "exante_12m.csv")
    ex_err = {m: np.mean(np.abs(ex["rzeczywiste"] - ex[m])) for m in
              ["Naive", "SARIMA", "LR", "LR-Δ", "XGBoost", "XGBoost-Δ", "LSTM-L", "LSTM-MO", variant, "LSTM (aplikacja)"]}
    cover = np.mean((ex["rzeczywiste"] >= ex.LSTM_lo_conf) & (ex["rzeczywiste"] <= ex.LSTM_hi_conf)) * 100
    pd.Series(ex_err, name="MAE_ex_ante_12m").to_csv(OUT / "exante_mae.csv")

    # --- 3. XAI na danych holdout: IG dla prognozy 1-krokowej z każdego punktu startowego ---
    A = []
    for f in lstm_models[variant]:
        a, comp = integrated_gradients(f, f.input_tensor(ye, origins))
        A.append(a[:, :, 0])
    A = np.mean(A, axis=0)
    w = A.shape[1]
    ig = pd.DataFrame(A[:, ::-1], index=ye.index[origins + 1], columns=[f"lag_{l}" for l in range(1, w + 1)])
    ig.index.name = "target_date"
    ig.to_csv(OUT / "ig_holdout.csv")

    # --- wykresy ---
    C = {"Naive": "#7f8c8d", "SARIMA": "#2980b9", "XGBoost": "#c0392b", "LR-Δ": "#16a085", "LSTM-L": "#8e44ad",
         variant: "#27ae60", "LSTM-MO": "#1abc9c", "XGBoost-Δ": "#e67e22"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # (a) ex ante
    fig, ax = plt.subplots(figsize=(10, 4.3))
    hist = ye.loc["2024-06":]
    ax.plot(hist.index, hist, color="black", lw=2.2, label="CPI rzeczywiste (od 11.2025 – dane nowe)")
    ax.axvspan(ho.index[0] - pd.Timedelta(days=15), ho.index[-1] + pd.Timedelta(days=15), color="#f4f6f8")
    ax.fill_between(ex.index, ex.LSTM_lo_conf, ex.LSTM_hi_conf, color=C[variant], alpha=.12,
                    label=f"{variant} – przedział 95% (konformalny)")
    for m in ["Naive", "SARIMA", "XGBoost", "LR-Δ", variant]:
        ax.plot(ex.index, ex[m], label=f"{m} (prognoza z 10.2025)", color=C[m], lw=1.6,
                ls="-" if m == variant else "--")
    ax.set_ylim(97, 109)
    ax.set(title="Ocena ex ante: prognoza na 12 miesięcy wykonana w październiku 2025", ylabel="CPI (r/r)")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_exante.png", dpi=160)
    plt.close(fig)
    # (b) rolling h=1
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(ho.index, ho, color="black", lw=2.2, marker="o", ms=4, label="CPI rzeczywiste")
    g = avg[(avg.h == 1) & (avg.model == variant)]
    ax.fill_between(g.target_date, g.lo_conf, g.hi_conf, color=C[variant], alpha=.12, label="95% konformalny")
    for m in ["Naive", "SARIMA", "XGBoost", "LR-Δ", variant]:
        g = avg[(avg.h == 1) & (avg.model == m)]
        ax.plot(g.target_date, g.y_pred, color=C[m], lw=1.5, ls="-" if m == variant else "--", label=m)
    ax.annotate("szok paliwowy\n(03.2026)", xy=(pd.Timestamp("2026-03-01"), 103.0), xytext=(pd.Timestamp("2025-12-15"), 104.3),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set(title="Prognozy 1-miesięczne na nowych danych (model zamrożony, 11.2025 – 10.2026)", ylabel="CPI (r/r)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_holdout_h1.png", dpi=160)
    plt.close(fig)
    # (c) IG na holdout
    fig, ax = plt.subplots(figsize=(10, 3.6))
    lim = np.abs(ig.values).max()
    im = ax.imshow(ig.values.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(ig)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in ig.index], rotation=45, fontsize=8)
    ax.set_yticks(range(w))
    ax.set_yticklabels(range(1, w + 1), fontsize=8)
    ax.set(ylabel="opóźnienie [mies.]", title=f"Integrated Gradients – {variant}, prognozy na miesiące 11.2025 – 10.2026")
    fig.colorbar(im, ax=ax, label="wkład w prognozowaną zmianę [p.p.]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ig_holdout.png", dpi=160)
    plt.close(fig)

    print("\nMAE na danych holdout (rolling origin, LSTM: ensemble 5 ziaren):")
    print(met.pivot_table(index="model", columns="h", values="MAE").round(3).to_string())
    print("\nMAE ex ante (12 mies. z 10.2025):")
    print(pd.Series(ex_err).round(3).sort_values().to_string())
    print(f"Pokrycie przedziału konformalnego (ex ante, 12 mies.): {cover:.0f}%")
    (OUT / "summary.json").write_text(json.dumps({"variant": variant, "exante_coverage_conf": cover,
                                                  "n_holdout": len(ho)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
