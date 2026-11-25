"""
Wykresy do sprawozdań projektu AI (połączone dawne report_figures.py i report_figures_km3.py).
    python report_figures.py          # wszystkie wykresy
    python report_figures.py km2      # tylko kamień milowy 2
    python report_figures.py km3      # tylko kamień milowy 3 (wymaga final_evaluation.py)
Wyniki -> results/report_figures/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_utils import load_series, HORIZONS


def figures_km2():
    """Kamień milowy 2: schemat, prognozy, MAE(h), ablacje E4-E6."""
    RES = Path("results")
    OUT = RES / "report_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    C = {"Naive": "#7f8c8d", "SARIMA": "#2980b9", "XGBoost": "#c0392b", "XGBoost-Δ": "#e67e22", "LR": "#16a085",
         "LSTM-L": "#8e44ad", "LSTM-Δ": "#27ae60", "LSTM-Δ+kal": "#82c91e", "LSTM-MO": "#1abc9c"}

    y = load_series()
    fc = pd.read_csv(RES / "forecasts.csv", parse_dates=["origin", "target_date"])
    summ = pd.read_csv(RES / "metrics_summary.csv")
    seed = pd.read_csv(RES / "metrics_per_seed.csv")
    S = summ[summ.period == "całość"]

    # 1. Schemat rozwiązania
    fig, ax = plt.subplots(figsize=(10, 2.3))
    ax.axis("off")
    boxes = [("Dane NBP\n(bazowa.xlsx)\nETL: data_source.py", 0.02), ("Okna przesuwne w=12/24\ncel: poziom lub Δy\nskalowanie (tylko trening)", 0.215),
             ("Model LSTM (PyTorch)\nLSTM → Dropout → Dense\nearly stopping na walidacji", 0.41),
             ("Ewaluacja rolling origin\nh = 1…12, podziały A/B\nMAE, RMSE, MASE, DA, DM", 0.605),
             ("Aplikacja Streamlit\nzakładka LSTM & XAI\nprognoza 12 mies.", 0.80)]
    for txt, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.28), 0.175, 0.68, fc="#eef5ee", ec="#27ae60", lw=1.5))
        ax.text(x + 0.0875, 0.62, txt, ha="center", va="center", fontsize=8.5)
    for _, x in boxes[:-1]:
        ax.annotate("", xy=(x + 0.195, 0.62), xytext=(x + 0.177, 0.62), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.06, "Modele bazowe w tym samym protokole: Naive, Seasonal Naive, LR, SARIMA, XGBoost, RF, k-NN, SVR "
            "(+ warianty Δ dla XGBoost i LR)", ha="center", fontsize=8.5, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_schemat.png", dpi=160)
    plt.close(fig)

    # 2. Prognozy h=1, Podział B (zbliżenie)
    for split, start in [("B", "2021-06"), ("A", "2020-06")]:
        g = fc[(fc.split == split) & (fc.h == 1)]
        avg = g.groupby(["model", "target_date"])["y_pred"].mean().unstack(0)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        yy = y.loc[start:]
        ax.plot(yy.index, yy, color="black", lw=2.2, label="CPI rzeczywiste")
        for m in ["Naive", "XGBoost", "LSTM-L", "LSTM-Δ", "LSTM-MO"]:
            a = avg[m].loc[start:]
            ax.plot(a.index, a, label=m, color=C[m], lw=1.4, ls="--" if m in ("Naive", "XGBoost") else "-")
        if split == "B":
            ax.axvline(pd.Timestamp("2023-01-01"), color="gray", ls=":")
            ax.text(pd.Timestamp("2023-01-20"), 106.5, "początek testu →", fontsize=8, color="gray")
        tr_max = y.loc[: (pd.Timestamp("2022-12-01") if split == "B" else pd.Timestamp("2014-12-01"))].max()
        ax.axhline(tr_max, color="gray", lw=.8, ls="-.")
        ax.text(yy.index[-30] if split == "A" else yy.index[0], tr_max - 0.9 if split == "A" else tr_max + 0.4, f"maksimum zbioru treningowego = {tr_max}", fontsize=8, color="gray")
        ax.set(ylabel="CPI (indeks, r/r)", title=f"Prognozy 1-miesięczne (rolling origin) – Podział {split}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_prognozy_h1_{split}.png", dpi=160)
        plt.close(fig)

    # 3. MAE vs h – oba podziały
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ["A", "B"]):
        for m in ["Naive", "SARIMA", "LR", "XGBoost", "XGBoost-Δ", "LSTM-L", "LSTM-Δ", "LSTM-MO"]:
            r = S[(S.split == split) & (S.model == m)].sort_values("h")
            ax.errorbar(r.h, r.MAE, yerr=r.MAE_std, marker="o", ms=4, capsize=3, label=m, color=C[m],
                        ls="-" if m.startswith("LSTM") else "--")
        ax.set(title=f"Podział {split}", xlabel="horyzont h [mies.]", ylabel="MAE [p.p.]")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=.3)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mae_vs_h.png", dpi=160)
    plt.close(fig)

    # 4. E4 – ablacja celu (okres ekstrapolacji 2021-2023, Podział A)
    P = summ[(summ.split == "A") & (summ.period == "ekstrapolacja 2021-2023")]
    models = ["LSTM-L", "LSTM-Δ", "XGBoost", "XGBoost-Δ"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.2
    for k, m in enumerate(models):
        r = P[P.model == m].set_index("h").loc[HORIZONS]
        ax.bar(np.arange(4) + (k - 1.5) * w, r.MAE, w, yerr=r.MAE_std, label=m, color=C[m], capsize=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set(ylabel="MAE [p.p.]", title="E4: poziom vs przyrost (Δy) – okres 2021-2023, Podział A")
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_E4_ablacja.png", dpi=160)
    plt.close(fig)

    # 5. E5 – rekurencja vs multi-output
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, split in zip(axes, ["A", "B"]):
        for k, m in enumerate(["LSTM-Δ+kal", "LSTM-MO"]):
            r = S[(S.split == split) & (S.model == m)].set_index("h").loc[HORIZONS]
            ax.bar(np.arange(4) + (k - 0.5) * 0.35, r.MAE, 0.35, yerr=r.MAE_std, capsize=2, color=C[m],
                   label="rekurencyjny (LSTM-Δ+kal)" if k == 0 else "bezpośredni (LSTM-MO)")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"h={h}" for h in HORIZONS])
        ax.set(title=f"Podział {split}", ylabel="MAE [p.p.]")
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("E5: strategia wielokrokowa")
    fig.tight_layout()
    fig.savefig(OUT / "fig_E5_strategia.png", dpi=160)
    plt.close(fig)

    # 6. E6 – stabilność między ziarnami
    sd = seed[(seed.period == "całość") & seed.model.str.startswith("LSTM")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, split in zip(axes, ["A", "B"]):
        data, labels = [], []
        for m in ["LSTM-L", "LSTM-Δ", "LSTM-Δ+kal", "LSTM-MO"]:
            data.append(sd[(sd.split == split) & (sd.model == m) & (sd.h == 1)].MAE.values)
            labels.append(m)
        ax.boxplot(data)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(labels, fontsize=8)
        for i, dd in enumerate(data):
            ax.scatter(np.full(len(dd), i + 1) + np.random.uniform(-.08, .08, len(dd)), dd, s=12, color="k", zorder=3)
        nv = S[(S.split == split) & (S.model == "Naive") & (S.h == 1)].MAE.iloc[0]
        xg = S[(S.split == split) & (S.model == "XGBoost") & (S.h == 1)].MAE.iloc[0]
        ax.axhline(nv, color=C["Naive"], ls="--", lw=1, label=f"Naive ({nv:.3f})")
        ax.axhline(xg, color=C["XGBoost"], ls="--", lw=1, label=f"XGBoost ({xg:.3f})")
        ax.set(title=f"Podział {split}, h=1 (5 ziaren)", ylabel="MAE [p.p.]")
        ax.legend(fontsize=7)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_E6_ziarna.png", dpi=160)
    plt.close(fig)
    print("Zapisano wykresy w", OUT)



def figures_km3():
    """Kamień milowy 3: XAI, obciążenie, kalibracja niepewności, wyjaśnienia lokalne."""
    RES = Path("results")
    X = RES / "xai"
    OUT = RES / "report_figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    C = {"Naive": "#7f8c8d", "SARIMA": "#2980b9", "XGBoost": "#c0392b", "XGBoost-Δ": "#e67e22", "LR-Δ": "#16a085",
         "LSTM-L": "#8e44ad", "LSTM-Δ": "#27ae60", "LSTM-MO": "#1abc9c"}

    imp = pd.read_csv(X / "importance.csv")

    # 1. Ważność opóźnień: LSTM-Δ (IG + okluzja) vs XGBoost (SHAP)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    lags = np.arange(1, 13)
    for k, sp in enumerate("AB"):
        g = imp[(imp.split == sp) & (imp.model == "LSTM-Δ") & (imp.method == "IG |atrybucja|")].copy()
        g["lag"] = g.lag.astype(int)
        g = g.set_index("lag").loc[lags]
        axes[0].bar(lags + (k - 0.5) * 0.38, g.importance, 0.38, label=f"Podział {sp}",
                    color=["#27ae60", "#a9dfbf"][k])
        o = imp[(imp.split == sp) & (imp.model == "LSTM-Δ") & (imp.method == "Okluzja ΔMAE")].copy()
        o["lag"] = o.lag.astype(int)
        o = o.set_index("lag").loc[lags]
        axes[1].bar(lags + (k - 0.5) * 0.38, o.importance, 0.38, label=f"Podział {sp}", color=["#2c3e50", "#aab7b8"][k])
    axes[0].set(title="LSTM-Δ: Integrated Gradients", xlabel="opóźnienie [mies.]", ylabel="średni |wkład| [p.p.]")
    axes[1].set(title="LSTM-Δ: okluzja (wzrost MAE, h=1)", xlabel="opóźnienie [mies.]", ylabel="ΔMAE [p.p.]")
    axes[1].axhline(0, color="black", lw=.6)
    for ax in axes[:2]:
        ax.set_xticks(lags)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3, axis="y")
    feats = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll_mean_3", "roll_mean_6", "month"]
    for k, sp in enumerate("AB"):
        s = imp[(imp.split == sp) & (imp.model == "XGBoost")].set_index("lag").loc[feats]
        axes[2].barh(np.arange(len(feats)) + (k - 0.5) * 0.38, s.importance, 0.38, label=f"Podział {sp}",
                     color=["#c0392b", "#f5b7b1"][k])
    axes[2].set_yticks(range(len(feats)))
    axes[2].set_yticklabels(feats, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set(title="XGBoost: TreeSHAP", xlabel="średni |SHAP| [p.p.]")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "km3_waznosc_opoznien.png", dpi=160)
    plt.close(fig)

    # 2. Mapa IG w czasie (Podział A, LSTM-Δ), 2019-2025
    heat = pd.read_csv(X / "ig_heatmap_A_LSTM-Δ.csv", index_col=0, parse_dates=True).loc["2019":]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    lim = np.percentile(np.abs(heat.values), 99)
    im = ax.imshow(heat.values.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ticks = [i for i, d in enumerate(heat.index) if d.month == 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([heat.index[i].year for i in ticks])
    ax.set_yticks(range(12))
    ax.set_yticklabels(range(1, 13), fontsize=8)
    ax.set(ylabel="opóźnienie [mies.]", title="Integrated Gradients w czasie – LSTM-Δ, Podział A (czerwony: podnosi prognozę zmiany CPI)")
    fig.colorbar(im, ax=ax, label="wkład [p.p.]")
    fig.tight_layout()
    fig.savefig(OUT / "km3_ig_mapa.png", dpi=160)
    plt.close(fig)

    # 3. Obciążenie w podokresach (h=1, Podział A)
    b = pd.read_csv(X / "bias_by_period.csv")
    b = b[(b.split == "A") & (b.h == 1)]
    per = ["stabilny 2015-2019", "szok 2020-02.2023", "dezinflacja 03.2023-"]
    models = ["Naive", "SARIMA", "XGBoost-Δ", "LSTM-L", "LSTM-Δ"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, m in enumerate(models):
        v = b[b.model == m].set_index("period").loc[per].bias
        ax.bar(np.arange(3) + (k - 2) * 0.16, v, 0.16, label=m, color=C[m])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["stabilny 2015–2019", "szok 2020–02.2023", "dezinflacja 03.2023–10.2025"])
    ax.set(ylabel="średni błąd (rzecz. − prognoza) [p.p.]",
           title="Obciążenie prognoz 1-miesięcznych (Podział A): > 0 model zaniża, < 0 zawyża")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "km3_obciazenie.png", dpi=160)
    plt.close(fig)

    # 4. Kalibracja niepewności
    u = pd.read_csv(X / "uncertainty.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, sp in zip(axes, "AB"):
        for (m, meth), lab, col, ls in [(("LSTM-Δ", "MC-dropout"), "LSTM-Δ MC-dropout", "#27ae60", ":"),
                                        (("LSTM-Δ", "konformalny (walidacja)"), "LSTM-Δ konformalny", "#27ae60", "-"),
                                        (("SARIMA", "analityczny"), "SARIMA (analityczny)", "#2980b9", "--")]:
            g = u[(u.split == sp) & (u.model == m) & (u.method == meth)].sort_values("h")
            ax.plot(g.h, g.coverage95, marker="o", label=lab, color=col, ls=ls)
        ax.axhline(95, color="black", lw=1, ls="-.", label="poziom nominalny 95%")
        ax.set_ylim(-3, 100)
        ax.set(title=f"Podział {sp}", xlabel="horyzont h", ylabel="pokrycie [%]")
        ax.set_xticks([1, 3, 6, 12])
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Empiryczne pokrycie przedziałów 95% na zbiorze testowym")
    fig.tight_layout()
    fig.savefig(OUT / "km3_kalibracja.png", dpi=160)
    plt.close(fig)

    # 5. Wyjaśnienia lokalne na danych holdout: prognoza na 03.2026 (szok) i 04.2026 (reakcja)
    ig = pd.read_csv(RES / "final" / "ig_holdout.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, d, ttl in [(axes[0], "2026-03-01", "Prognoza na 03.2026 (z danych do 02.2026)"),
                       (axes[1], "2026-04-01", "Prognoza na 04.2026 (z danych do 03.2026)")]:
        row = ig.loc[d]
        months = pd.date_range(end=pd.Timestamp(d) - pd.DateOffset(months=1), periods=12, freq="MS")[::-1]
        cols = ["#c0392b" if v > 0 else "#2980b9" for v in row.values]
        ax.bar(range(12), row.values, color=cols)
        ax.set_xticks(range(12))
        ax.set_xticklabels([m.strftime("%y-%m") for m in months], rotation=60, fontsize=7.5)
        ax.axhline(0, color="black", lw=.6)
        ax.set(title=ttl, xlabel="miesiąc wejściowy (od najnowszego)")
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("wkład w prognozowaną zmianę [p.p.]")
    fig.tight_layout()
    fig.savefig(OUT / "km3_wyjasnienie_lokalne.png", dpi=160)
    plt.close(fig)
    print("Zapisano wykresy KM3 w", OUT)



if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "km2"):
        figures_km2()
    if which in ("all", "km3"):
        figures_km3()
