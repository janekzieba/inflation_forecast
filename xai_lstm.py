"""
Kamień milowy 3: warstwa Explainable AI (XAI) + analiza błędów.

1. Integrated Gradients (IG) dla LSTM – wkład każdego z ostatnich w miesięcy w prognozę (w p.p.).
2. Okluzja opóźnień (model-agnostyczna kontrola IG) – wzrost MAE po "wyzerowaniu" danego miesiąca.
3. SHAP dla XGBoost (TreeSHAP wbudowany w xgboost, pred_contribs=True) – porównanie z LSTM.
4. Analiza błędów: największe błędy, podokresy, punkty zwrotne, ekstrapolacja poza zakres treningu,
   zależność błędu od zmienności.
5. Niepewność: MC-dropout – pokrycie przedziałów i związek szerokości przedziału z błędem.

Uruchomienie (po experiments.py):
    python xai_lstm.py
Wyniki -> results/xai/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval_utils import SPLITS, PERIODS, load_series, origins_for_split, build_train_matrix, \
    features_from_buffer, FEATURES, make_ml_models, filter_period
from lstm_model import LSTMForecaster

RES = Path("results")
OUT = RES / "xai"


# ------------------------------------------------------------------
# 1. Integrated Gradients
# ------------------------------------------------------------------
def integrated_gradients(f: LSTMForecaster, X: torch.Tensor, out_idx: int = 0, steps: int = 64):
    """IG względem wejścia X (n, w, c). Linia bazowa = 0 w przestrzeni przeskalowanej
    (czyli średnia z treningu – dla wariantu Δ: 'typowa' miesięczna zmiana CPI).
    Zwraca atrybucje (n, w, c) w jednostkach wyjścia modelu (p.p.) oraz błąd zupełności."""
    net = f.net
    net.eval()
    x0 = torch.zeros_like(X)
    total = torch.zeros_like(X)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (x0 + a * (X - x0)).requires_grad_(True)
        out = net(xi)[:, out_idx].sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    attr = (X - x0) * total / steps
    with torch.no_grad():
        delta = net(X)[:, out_idx] - net(x0)[:, out_idx]
    completeness = (attr.sum(dim=(1, 2)) - delta).abs().mean().item()
    return attr.detach().numpy() * f.sd, completeness * f.sd


def occlusion(f: LSTMForecaster, y: pd.Series, origins: np.ndarray):
    """Dla każdego opóźnienia k: zastąp wartość wejściową średnią treningową i zmierz zmianę prognozy 1-krokowej."""
    X = f.input_tensor(y, origins)
    with torch.no_grad():
        base = f.net(X)[:, 0].numpy() * f.sd
    true = y.values[origins + 1]
    level0 = y.values[origins]
    add = f.mu + (level0 if f.cfg.target == "delta" else 0)
    pred_base = base + add
    mae_base = np.mean(np.abs(true - pred_base))
    rows = []
    w = X.shape[1]
    for k in range(w):
        Xo = X.clone()
        Xo[:, w - 1 - k, 0] = 0.0          # opóźnienie k+1 (k=0 -> ostatni znany miesiąc)
        with torch.no_grad():
            p = f.net(Xo)[:, 0].numpy() * f.sd
        pred = p + add
        rows.append(dict(lag=k + 1, mean_abs_change=np.mean(np.abs(p - base)),
                         MAE_increase=np.mean(np.abs(true - pred)) - mae_base))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 3. SHAP dla XGBoost
# ------------------------------------------------------------------
def xgb_shap(y: pd.Series, split: str):
    import xgboost as xgb
    split_pos = y.index.get_loc(SPLITS[split])
    X, Y = build_train_matrix(y, split_pos)
    model = make_ml_models()["XGBoost"].fit(X, Y)
    origins = origins_for_split(y, split)
    v = y.values
    buf = np.stack([v[i - 11:i + 1] for i in origins])
    months = np.array([(y.index[i] + pd.DateOffset(months=1)).month for i in origins])
    Xt = features_from_buffer(buf, months)
    contrib = model.get_booster().predict(xgb.DMatrix(Xt, feature_names=FEATURES), pred_contribs=True)
    shap = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=y.index[origins + 1])
    return shap


# ------------------------------------------------------------------
def load_models(split, variant):
    paths = sorted((RES / "models").glob(f"{split}_{variant}_seed*.pt"))
    return [LSTMForecaster.load(p) for p in paths]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y = load_series()
    best = json.loads((RES / "best_lstm.json").read_text())
    fc = pd.read_csv(RES / "forecasts.csv", parse_dates=["origin", "target_date"])
    md = ["# Raport XAI i analiza błędów (wygenerowane przez xai_lstm.py)\n"]

    imp_rows, heat_frames = [], {}
    for split in SPLITS:
        origins = origins_for_split(y, split)
        variants = sorted({best[split], "LSTM-L", "LSTM-Δ"})
        for variant in variants:
            models = load_models(split, variant)
            if not models:
                continue
            attrs, comp = [], []
            occ = []
            for f in models:
                X = f.input_tensor(y, origins)
                a, c = integrated_gradients(f, X)
                attrs.append(a[:, :, 0])  # kanał szeregu (bez kalendarza)
                comp.append(c)
                occ.append(occlusion(f, y, origins))
            A = np.mean(attrs, axis=0)           # (n, w) – uśrednione po ziarnach
            w = A.shape[1]
            lags = np.arange(w, 0, -1)          # kolumna 0 = najstarszy miesiąc
            heat = pd.DataFrame(A, index=y.index[origins + 1], columns=[f"lag_{l}" for l in lags])
            heat = heat[[f"lag_{l}" for l in range(1, w + 1)]]
            heat.to_csv(OUT / f"ig_heatmap_{split}_{variant}.csv")
            heat_frames[(split, variant)] = heat
            occ_df = pd.concat(occ).groupby("lag").mean().reset_index()
            for l in range(1, w + 1):
                imp_rows.append(dict(split=split, model=variant, lag=l, method="IG |atrybucja|",
                                     importance=heat[f"lag_{l}"].abs().mean()))
                r = occ_df[occ_df.lag == l].iloc[0]
                imp_rows.append(dict(split=split, model=variant, lag=l, method="Okluzja ΔMAE",
                                     importance=r.MAE_increase))
            print(f"[{split}] {variant}: IG błąd zupełności = {np.mean(comp):.4f} p.p.")

        shap = xgb_shap(y, split)
        shap.to_csv(OUT / f"xgb_shap_{split}.csv")
        for feat in FEATURES:
            imp_rows.append(dict(split=split, model="XGBoost", lag=feat, method="SHAP |wkład|",
                                 importance=shap[feat].abs().mean()))

    imp = pd.DataFrame(imp_rows)
    imp.to_csv(OUT / "importance.csv", index=False)

    # --- wykresy ważności ---
    for split in SPLITS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        sub = imp[(imp.split == split) & (imp.method == "IG |atrybucja|")]
        for m, g in sub.groupby("model"):
            axes[0].plot(g.lag.astype(int), g.importance, marker="o", label=m)
        axes[0].set(title=f"LSTM – Integrated Gradients (Podział {split})", xlabel="opóźnienie [mies.]",
                    ylabel="średni |wkład| w prognozę [p.p.]")
        axes[0].grid(alpha=.3)
        axes[0].legend()
        sh = imp[(imp.split == split) & (imp.model == "XGBoost")].sort_values("importance")
        axes[1].barh(sh.lag.astype(str), sh.importance, color="tab:red")
        axes[1].set(title=f"XGBoost – SHAP (Podział {split})", xlabel="średni |SHAP| [p.p.]")
        fig.tight_layout()
        fig.savefig(OUT / f"importance_{split}.png", dpi=130)
        plt.close(fig)

    for (split, variant), heat in heat_frames.items():
        if variant != best[split] or split != "A":
            continue
        fig, ax = plt.subplots(figsize=(12, 4.5))
        lim = np.percentile(np.abs(heat.values), 98)
        im = ax.imshow(heat.values.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(range(heat.shape[1]))
        ax.set_yticklabels(range(1, heat.shape[1] + 1), fontsize=6)
        ticks = np.arange(0, len(heat), 12)
        ax.set_xticks(ticks)
        ax.set_xticklabels([heat.index[i].strftime("%Y") for i in ticks])
        ax.set(ylabel="opóźnienie [mies.]", title=f"IG w czasie – {variant}, Podział {split} (czerwony = podnosi prognozę)")
        fig.colorbar(im, ax=ax, label="wkład [p.p.]")
        fig.tight_layout()
        fig.savefig(OUT / f"ig_heatmap_{split}.png", dpi=130)
        plt.close(fig)

    md.append("## 1. Które miesiące historii decydują o prognozie?\n")
    for split in SPLITS:
        b = best[split]
        s = imp[(imp.split == split) & (imp.model == b) & (imp.method == "IG |atrybucja|")]
        s = s.sort_values("importance", ascending=False).head(5)
        sh = imp[(imp.split == split) & (imp.model == "XGBoost")].sort_values("importance", ascending=False).head(3)
        md.append(f"- Podział {split}, {b}: najważniejsze opóźnienia wg IG: " +
                  ", ".join(f"lag {r.lag} ({r.importance:.3f})" for r in s.itertuples()) +
                  ". XGBoost (SHAP): " + ", ".join(f"{r.lag} ({r.importance:.3f})" for r in sh.itertuples()) + ".")
    md.append("")

    # ------------------------------------------------------------------
    # 4. Analiza błędów
    # ------------------------------------------------------------------
    models_err = ["Naive", "SARIMA", "XGBoost", "XGBoost-Δ", "LSTM-L", "LSTM-Δ", "LSTM-Δ+kal", "LSTM-MO"]
    avg = fc.groupby(["split", "model", "h", "origin", "target_date"], as_index=False)[
        ["y_origin", "y_true", "y_pred", "lo", "hi", "lo_conf", "hi_conf"]].mean()
    avg["err"] = avg.y_true - avg.y_pred
    avg.to_csv(OUT / "errors_ensemble.csv", index=False)

    # 4a: obciążenie (średni błąd ze znakiem) w podokresach – czy model "spóźnia się"?
    rows = []
    for (split, m, h), g in avg[avg.model.isin(models_err)].groupby(["split", "model", "h"]):
        for p in PERIODS:
            gp = filter_period(g, p)
            if len(gp) >= 3:
                rows.append(dict(split=split, model=m, h=h, period=p, bias=gp.err.mean(),
                                 MAE=gp.err.abs().mean(), n=len(gp)))
    bias = pd.DataFrame(rows)
    bias.to_csv(OUT / "bias_by_period.csv", index=False)

    # 4b: największe błędy najlepszego LSTM
    tops = []
    for split in SPLITS:
        g = avg[(avg.split == split) & (avg.model == best[split]) & (avg.h.isin([1, 3]))]
        t = g.reindex(g.err.abs().sort_values(ascending=False).index).head(10)
        tops.append(t.assign(model=best[split]))
    top = pd.concat(tops)
    top.to_csv(OUT / "top_errors.csv", index=False)

    # 4c: punkty zwrotne – trafność kierunku w miesiącach, w których zmienia się znak 3-mies. zmiany CPI
    d3 = y.diff(3)
    turning = (np.sign(d3) != np.sign(d3.shift(1))) & d3.notna() & d3.shift(1).notna()
    tp_dates = set(y.index[turning.values])
    rows = []
    for (split, m, h), g in avg[avg.model.isin(models_err) & avg.h.isin([1, 3])].groupby(["split", "model", "h"]):
        is_tp = g.target_date.isin(tp_dates)
        hit = np.sign(g.y_true - g.y_origin) == np.sign(g.y_pred - g.y_origin)
        rows.append(dict(split=split, model=m, h=h, DA_turning=hit[is_tp].mean() * 100,
                         DA_other=hit[~is_tp].mean() * 100, MAE_turning=g.err[is_tp].abs().mean(),
                         MAE_other=g.err[~is_tp].abs().mean(), n_turning=int(is_tp.sum())))
    tp = pd.DataFrame(rows)
    tp.to_csv(OUT / "turning_points.csv", index=False)

    # 4d: ekstrapolacja – Podział A, CPI powyżej maksimum treningu
    y_max = y.loc[: SPLITS["A"] - pd.DateOffset(months=1)].max()
    rows = []
    for m in models_err:
        g = avg[(avg.split == "A") & (avg.model == m) & (avg.h == 1)]
        above = g.y_true > y_max
        rows.append(dict(model=m, train_max=y_max, n_above=int(above.sum()),
                         share_pred_above_max=(g.y_pred[above] > y_max).mean() * 100,
                         max_pred=g.y_pred.max(), MAE_above=g.err[above].abs().mean(),
                         MAE_within=g.err[~above].abs().mean()))
    ext = pd.DataFrame(rows)
    ext.to_csv(OUT / "extrapolation.csv", index=False)

    # 4e: błąd a zmienność (odchylenie std Δy z 12 mies. przed punktem startowym)
    vol = y.diff().rolling(12).std()
    rows = []
    for (split, m), g in avg[avg.model.isin(models_err) & (avg.h == 1)].groupby(["split", "model"]):
        v = vol.reindex(g.origin).values
        rows.append(dict(split=split, model=m, corr_abs_err_volatility=np.corrcoef(v, g.err.abs())[0, 1]))
    pd.DataFrame(rows).to_csv(OUT / "error_vs_volatility.csv", index=False)

    # 5. MC-dropout: czy model "wie, kiedy nie wie"?
    rows = []
    for split in SPLITS:
        g = avg[(avg.split == split) & avg.model.str.startswith("LSTM")].dropna(subset=["lo"])
        for (m, h), gg in g.groupby(["model", "h"]):
            if h not in (1, 3, 6, 12):
                continue
            width = gg.hi - gg.lo
            rows.append(dict(split=split, model=m, h=h, method="MC-dropout",
                             coverage95=((gg.y_true >= gg.lo) & (gg.y_true <= gg.hi)).mean() * 100,
                             mean_width=width.mean(), corr_width_abs_err=np.corrcoef(width, gg.err.abs())[0, 1]))
            rows.append(dict(split=split, model=m, h=h, method="konformalny (walidacja)",
                             coverage95=((gg.y_true >= gg.lo_conf) & (gg.y_true <= gg.hi_conf)).mean() * 100,
                             mean_width=(gg.hi_conf - gg.lo_conf).mean(), corr_width_abs_err=np.nan))
        gs = avg[(avg.split == split) & (avg.model == "SARIMA")]
        for h in (1, 3, 6, 12):
            gg = gs[gs.h == h]
            width = gg.hi - gg.lo
            rows.append(dict(split=split, model="SARIMA", h=h, method="analityczny",
                             coverage95=((gg.y_true >= gg.lo) & (gg.y_true <= gg.hi)).mean() * 100,
                             mean_width=width.mean(), corr_width_abs_err=np.corrcoef(width, gg.err.abs())[0, 1]))
    unc = pd.DataFrame(rows)
    unc.to_csv(OUT / "uncertainty.csv", index=False)

    # wykres błędów w czasie
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for ax, split in zip(axes, SPLITS):
        for m in ["Naive", "XGBoost", "LSTM-L", best[split]]:
            g = avg[(avg.split == split) & (avg.model == m) & (avg.h == 1)]
            ax.plot(g.target_date, g.err, label=m, lw=1.2)
        ax.axhline(0, color="black", lw=.8)
        ax.set(title=f"Błąd prognozy h=1 (rzeczywiste − prognoza), Podział {split}", ylabel="p.p.")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "errors_over_time.png", dpi=130)
    plt.close(fig)

    # krzywe uczenia
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, SPLITS):
        for f in load_models(split, best[split])[:3]:
            ax.plot(f.history["train"], color="tab:blue", alpha=.6)
            ax.plot(f.history["val"], color="tab:orange", alpha=.6)
            ax.axvline(f.best_epoch, color="gray", ls=":", lw=.8)
        ax.set(title=f"Krzywe uczenia {best[split]} (niebieski: train, pomarańczowy: val) – {split}",
               xlabel="epoka", ylabel="strata (Huber)")
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=130)
    plt.close(fig)

    # --- raport tekstowy ---
    md.append("## 2. Gdzie i dlaczego model się myli?\n")
    md.append("### Obciążenie w podokresach (h=1, średni błąd ze znakiem; > 0 = model zaniża)\n")
    md.append(bias[(bias.h == 1)].pivot_table(index=["split", "model"], columns="period", values="bias")
              .round(3).to_markdown())
    md.append("\n### Punkty zwrotne (h=1)\n")
    md.append(tp[tp.h == 1].round(2).to_markdown(index=False))
    md.append("\n### Ekstrapolacja poza maksimum treningu (Podział A, max = %.1f)\n" % y_max)
    md.append(ext.round(3).to_markdown(index=False))
    md.append("\n### 10 największych błędów najlepszego LSTM (h ∈ {1,3})\n")
    md.append(top[["split", "model", "h", "origin", "target_date", "y_true", "y_pred", "err"]]
              .round(3).to_markdown(index=False))
    md.append("\n## 3. Niepewność (MC-dropout vs SARIMA)\n")
    md.append(unc.round(3).to_markdown(index=False))
    (OUT / "xai_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:8]))
    print(f"\n>>> Gotowe. Wyniki XAI w {OUT.resolve()}")


if __name__ == "__main__":
    main()
