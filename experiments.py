"""
Kamień milowy 2: realizacja eksperymentów E1-E6 (plan z sekcji 4.3).

Uruchomienie:
    python experiments.py            # pełny protokół (ok. 10-20 min na laptopie)
    python experiments.py --quick    # szybki test (mniejsza siatka, 2 ziarna) ~2-3 min

Wyniki -> folder results/
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_utils import (SPLITS, HORIZONS, H_MAX, load_series, origins_for_split, to_long,
                        metrics_table, summarize, dm_table, make_ml_models, run_ml_model, run_naive,
                        run_seasonal_naive, run_sarima)
from lstm_model import LSTMForecaster, make_config, VARIANTS

warnings.filterwarnings("ignore")
OUT = Path("results")
MODELS_DIR = OUT / "models"
BASELINES = ["Naive", "SeasonalNaive", "LR", "SARIMA", "XGBoost", "RandomForest", "kNN", "SVR"]
LSTM_NAMES = list(VARIANTS)


def validation_errors(f: LSTMForecaster, y: pd.Series, train_end: int) -> dict:
    """Walidacja krocząca wewnątrz zbioru treningowego: prognozy z punktów startowych w ostatnich 24 mies.
    treningu (tylko cele < początek testu). Zwraca {h: wektor |błędów|} dla h = 1..12."""
    vs = train_end - f.cfg.val_months
    origins = np.arange(vs - 1, train_end - 1)
    p = f.forecast(y, origins)["point"]
    errs = {}
    for h in range(1, H_MAX + 1):
        idx = [r for r, i in enumerate(origins) if i + h < train_end]
        errs[h] = np.abs(y.values[origins[idx] + h] - p[idx, h - 1])
    return errs


def validation_score(f: LSTMForecaster, y: pd.Series, train_end: int) -> float:
    """Średnie MAE walidacji kroczącej po horyzontach {1,3,6,12} – kryterium wyboru hiperparametrów i wariantu."""
    e = validation_errors(f, y, train_end)
    return float(np.mean([e[h].mean() for h in HORIZONS]))


def conformal_bounds(point: np.ndarray, errs: dict, alpha: float = 0.05):
    """Przedziały konformalne (split-conformal): ± kwantyl 95% |błędów| walidacji dla danego h."""
    q = np.array([np.quantile(errs[h], 1 - alpha) for h in range(1, point.shape[1] + 1)])
    return point - q, point + q


# ------------------------------------------------------------------
def run_baselines(y, split, log):
    fcs, timing = [], []
    p, o = run_naive(y, split)
    fcs.append(to_long("Naive", 0, y, o, p))
    p, o = run_seasonal_naive(y, split)
    fcs.append(to_long("SeasonalNaive", 0, y, o, p))
    p, lo, hi, o, tt, ti = run_sarima(y, split)
    fcs.append(to_long("SARIMA", 0, y, o, p, lo, hi))
    timing.append(dict(split=split, model="SARIMA", train_s=tt, infer_s=ti))
    log(f"  SARIMA gotowa ({tt + ti:.1f}s)")
    for name, model in make_ml_models().items():
        p, o, tt, ti, _ = run_ml_model(name, model, y, split, "level")
        fcs.append(to_long(name, 0, y, o, p))
        timing.append(dict(split=split, model=name, train_s=tt, infer_s=ti))
    # E4: ablacja celu dla XGBoost (i LR jako kontrola)
    for name in ["XGBoost", "LR"]:
        model = make_ml_models()[name]
        p, o, tt, ti, _ = run_ml_model(name, model, y, split, "delta")
        fcs.append(to_long(f"{name}-Δ", 0, y, o, p))
        timing.append(dict(split=split, model=f"{name}-Δ", train_s=tt, infer_s=ti))
    log("  modele ML gotowe")
    return pd.concat(fcs), timing


def grid_search(y, split, grid, seeds, log):
    """E6: przegląd siatki dla wariantu LSTM-Δ, ocena walidacją kroczącą (bez użycia zbioru testowego)."""
    train_end = y.index.get_loc(SPLITS[split])
    rows = []
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    for c, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        for seed in seeds:
            f = LSTMForecaster(make_config("LSTM-Δ", params, seed=seed)).fit(y, train_end)
            rows.append(dict(split=split, **params, seed=seed, val_MAE=validation_score(f, y, train_end),
                             best_epoch=f.best_epoch, train_s=f.train_time))
        log(f"  siatka {c + 1}/{len(combos)}: {params} -> val MAE "
            f"{np.mean([r['val_MAE'] for r in rows[-len(seeds):]]):.3f}")
    df = pd.DataFrame(rows)
    agg = df.groupby(keys)["val_MAE"].agg(["mean", "std"]).reset_index().sort_values("mean")
    best = agg.iloc[0][keys].to_dict()
    best = {k: (int(v) if k in ("window", "hidden") else float(v)) for k, v in best.items()}
    return df, best


def run_lstm(y, split, base, seeds, mc, log):
    train_end = y.index.get_loc(SPLITS[split])
    origins = origins_for_split(y, split)
    fcs, timing, vals = [], [], []
    for v in LSTM_NAMES:
        for seed in seeds:
            f = LSTMForecaster(make_config(v, base, seed=seed)).fit(y, train_end)
            t0 = time.time()
            out = f.forecast(y, origins, mc_samples=mc)
            ti = time.time() - t0
            errs = validation_errors(f, y, train_end)
            lc, hc = conformal_bounds(out["point"], errs)
            part = to_long(v, seed, y, origins, out["point"], out.get("lo"), out.get("hi"))
            cl = to_long(v, seed, y, origins, lc, lc, hc)
            part["lo_conf"], part["hi_conf"] = cl["lo"].values, cl["hi"].values
            fcs.append(part)
            timing.append(dict(split=split, model=v, seed=seed, train_s=f.train_time, infer_s=ti,
                               best_epoch=f.best_epoch))
            vals.append(dict(split=split, model=v, seed=seed,
                             val_MAE=float(np.mean([errs[h].mean() for h in HORIZONS]))))
            f.save(MODELS_DIR / f"{split}_{v}_seed{seed}.pt")
        log(f"  {v}: gotowe ({len(seeds)} ziaren)")
    return pd.concat(fcs), timing, pd.DataFrame(vals)


# ------------------------------------------------------------------
def evaluate_hypotheses(summ, dm, val_scores, best_cfg):
    L = []
    w = L.append
    S = summ[summ.period == "całość"]

    def mae(split, model, h, period="całość"):
        r = summ[(summ.split == split) & (summ.model == model) & (summ.h == h) & (summ.period == period)]
        return float(r["MAE"].iloc[0]) if len(r) else np.nan

    def mae_std(split, model, h):
        r = S[(S.split == split) & (S.model == model) & (S.h == h)]
        return float(r["MAE_std"].iloc[0]) if len(r) else np.nan

    best_lstm = {}
    for split in SPLITS:
        vs = val_scores[val_scores.split == split].groupby("model")["val_MAE"].mean()
        best_lstm[split] = vs.idxmin()

    w("# Weryfikacja hipotez (wygenerowane automatycznie przez experiments.py)\n")
    w("Najlepszy wariant LSTM wybrano na podstawie **walidacji kroczącej w zbiorze treningowym** "
      "(bez podglądania testu):")
    for split in SPLITS:
        w(f"- Podział {split}: **{best_lstm[split]}**, hiperparametry: `{json.dumps(best_cfg[split])}`")
    w("")

    # H1
    w("## H1: co najmniej jeden model uczony jest istotnie lepszy od Naive (DM, α = 0,05)\n")
    d = dm[(dm.reference == "Naive") & (dm.period == "całość")]
    for split in SPLITS:
        sig = d[(d.split == split) & (d.p_model_better < 0.05)]
        if len(sig):
            items = ", ".join(f"{r.model} (h={r.h}, p={r.p_model_better:.3f})" for r in sig.itertuples())
            w(f"- Podział {split}: **POTWIERDZONA** – {items}")
        else:
            w(f"- Podział {split}: **NIE POTWIERDZONA** – żaden model nie jest istotnie lepszy od Naive")
    w("")

    # H2
    w("## H2: w okresie zmiany reżimu (Podział B) LSTM ma niższe MAE niż XGBoost\n")
    w("Punkt odniesienia z pracy inżynierskiej: XGBoost MAE = 0,997 (prognoza 1-krokowa).\n")
    w("| h | " + " | ".join(LSTM_NAMES) + " | XGBoost | XGBoost-Δ | wynik (najlepszy LSTM) |")
    w("|---|" + "---|" * (len(LSTM_NAMES) + 3))
    for h in HORIZONS:
        vals = [mae("B", m, h) for m in LSTM_NAMES]
        xg = mae("B", "XGBoost", h)
        b = mae("B", best_lstm["B"], h)
        verdict = "TAK" if b < xg else "NIE"
        w(f"| {h} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {xg:.3f} | {mae('B', 'XGBoost-Δ', h):.3f} | {verdict} |")
    w("")

    # H3
    w("## H3: prognozowanie przyrostów zmniejsza błąd modeli nieliniowych w okresie 2021-2023\n")
    w("Okres 'ekstrapolacja 2021-2023', Podział A (wartości CPI poza zakresem treningu 2001-2014, max 107,4).\n")
    w("| h | LSTM-L | LSTM-Δ | XGBoost | XGBoost-Δ | LSTM: Δ lepszy? | XGB: Δ lepszy? |")
    w("|---|---|---|---|---|---|---|")
    for h in HORIZONS:
        a, b = mae("A", "LSTM-L", h, "ekstrapolacja 2021-2023"), mae("A", "LSTM-Δ", h, "ekstrapolacja 2021-2023")
        c, e = mae("A", "XGBoost", h, "ekstrapolacja 2021-2023"), mae("A", "XGBoost-Δ", h, "ekstrapolacja 2021-2023")
        w(f"| {h} | {a:.3f} | {b:.3f} | {c:.3f} | {e:.3f} | {'TAK' if b < a else 'NIE'} | {'TAK' if e < c else 'NIE'} |")
    w("")

    # H4
    w("## H4: przewaga LSTM nad modelami bazowymi rośnie z horyzontem\n")
    for split in SPLITS:
        bl = best_lstm[split]
        w(f"Podział {split} – {bl}: RelMAE (MAE / MAE Naive; < 1 = lepiej niż Naive)\n")
        w("| h | RelMAE LSTM | RelMAE SARIMA | RelMAE XGBoost | MAE najlepszego modelu bazowego | MAE LSTM |")
        w("|---|---|---|---|---|---|")
        gaps = []
        for h in HORIZONS:
            r = S[(S.split == split) & (S.h == h)].set_index("model")
            base = r.loc[[m for m in BASELINES if m in r.index], "MAE"]
            gaps.append(base.min() - r.loc[bl, "MAE"])
            w(f"| {h} | {r.loc[bl, 'RelMAE']:.3f} | {r.loc['SARIMA', 'RelMAE']:.3f} | {r.loc['XGBoost', 'RelMAE']:.3f} "
              f"| {base.min():.3f} ({base.idxmin()}) | {r.loc[bl, 'MAE']:.3f} |")
        rel = [S[(S.split == split) & (S.h == h) & (S.model == bl)]["RelMAE"].iloc[0] for h in HORIZONS]
        trend = np.polyfit(np.log(HORIZONS), rel, 1)[0]
        w(f"\nNachylenie RelMAE względem log(h): {trend:+.3f} -> "
          f"{'przewaga nad Naive ROŚNIE z h' if trend < 0 else 'przewaga nad Naive NIE rośnie z h'}.\n")
    w("")

    # Kryteria sukcesu
    w("## Kryteria sukcesu (sekcja 4.4)\n")
    for split in SPLITS:
        bl = best_lstm[split]
        d1 = dm[(dm.split == split) & (dm.model == bl) & (dm.reference == "Naive") & (dm.period == "całość")]
        ok1 = any((S[(S.split == split) & (S.model == bl) & (S.h == r.h)]["RelMAE"].iloc[0] < 1) and r.p_model_better < 0.05
                  for r in d1.itertuples())
        stds = [mae_std(split, bl, h) for h in HORIZONS]
        gaps = []
        for h in HORIZONS:
            r = S[(S.split == split) & (S.h == h)].set_index("model")
            base = r.loc[[m for m in BASELINES if m in r.index], "MAE"].min()
            gaps.append(abs(base - r.loc[bl, "MAE"]))
        ok3 = np.mean(stds) < np.mean(gaps)
        w(f"**Podział {split} ({bl})**")
        w(f"1. RelMAE < 1 i istotność DM vs Naive dla ≥1 horyzontu: **{'SPEŁNIONE' if ok1 else 'NIESPEŁNIONE'}**")
        if split == "B":
            ok2 = mae("B", bl, 1) < mae("B", "XGBoost", 1)
            w(f"2. MAE (h=1) niższe niż XGBoost: {mae('B', bl, 1):.3f} vs {mae('B', 'XGBoost', 1):.3f} -> "
              f"**{'SPEŁNIONE' if ok2 else 'NIESPEŁNIONE'}**")
        w(f"3. Stabilność: średnie std MAE po ziarnach = {np.mean(stds):.3f}, średnia |różnica| względem najlepszego "
          f"modelu bazowego = {np.mean(gaps):.3f} -> **{'SPEŁNIONE' if ok3 else 'NIESPEŁNIONE'}**\n")
    return "\n".join(L), best_lstm


# ------------------------------------------------------------------
def make_plots(summ, fc_all, grid_df, y, best_lstm):
    P = OUT / "plots"
    P.mkdir(parents=True, exist_ok=True)
    show = ["Naive", "SARIMA", "XGBoost", "XGBoost-Δ", "LR"] + LSTM_NAMES
    for split in SPLITS:
        S = summ[(summ.split == split) & (summ.period == "całość")]
        fig, ax = plt.subplots(figsize=(9, 5))
        for m in show:
            r = S[S.model == m].sort_values("h")
            if len(r):
                ls = "-" if m.startswith("LSTM") else "--"
                ax.errorbar(r.h, r.MAE, yerr=r.MAE_std, label=m, ls=ls, marker="o", capsize=3)
        ax.set(xlabel="horyzont h [mies.]", ylabel="MAE [p.p.]", title=f"MAE vs horyzont – Podział {split}")
        ax.set_xticks(HORIZONS)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(P / f"mae_vs_h_split{split}.png", dpi=130)
        plt.close(fig)

        for h in [1, 6]:
            fc = fc_all[(fc_all.split == split) & (fc_all.h == h)]
            avg = fc.groupby(["model", "target_date"])["y_pred"].mean().unstack(0)
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(y.loc["2012":].index, y.loc["2012":], color="black", lw=2, label="CPI rzeczywiste")
            for m in ["Naive", "SARIMA", "XGBoost", "LSTM-L", best_lstm[split]]:
                if m in avg:
                    ax.plot(avg.index, avg[m], label=m, lw=1.3)
            band = fc[fc.model == best_lstm[split]].groupby("target_date")[["lo", "hi"]].mean()
            ax.fill_between(band.index, band.lo, band.hi, alpha=.2, label=f"{best_lstm[split]} MC-dropout 95%")
            ax.axvline(SPLITS[split], color="gray", ls=":")
            ax.set(title=f"Prognozy h={h} – Podział {split}", ylabel="CPI (indeks)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(P / f"forecast_h{h}_split{split}.png", dpi=130)
            plt.close(fig)

    if grid_df is not None and len(grid_df):
        for split in SPLITS:
            g = grid_df[grid_df.split == split]
            if not len(g):
                continue
            g = g.assign(cfg=g.apply(lambda r: f"w{int(r.window)} u{int(r.hidden)} d{r.dropout} lr{r.lr}", axis=1))
            order = g.groupby("cfg")["val_MAE"].mean().sort_values().index
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot([g[g.cfg == c]["val_MAE"].values for c in order])
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order)
            ax.set(title=f"E6: wrażliwość LSTM-Δ na hiperparametry i ziarno (Podział {split})", ylabel="val MAE")
            plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
            fig.tight_layout()
            fig.savefig(P / f"grid_split{split}.png", dpi=130)
            plt.close(fig)


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="szybki przebieg testowy")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--mc", type=int, default=100, help="liczba przebiegów MC-dropout")
    ap.add_argument("--skip-grid", action="store_true", help="użyj zapisanych results/best_config.json")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    t_all = time.time()

    def log(msg):
        print(f"[{time.time() - t_all:6.1f}s] {msg}", flush=True)

    if args.quick:
        seeds = list(range(2))
        grid = dict(window=[12, 24], hidden=[32], dropout=[0.2], lr=[1e-3, 5e-3])
        grid_seeds = [0]
        mc = 30
    else:
        seeds = list(range(args.seeds))
        grid = dict(window=[12, 24], hidden=[32, 64], dropout=[0.1, 0.3], lr=[1e-3, 5e-3])
        grid_seeds = [0, 1, 2]
        mc = args.mc

    y = load_series()
    log(f"Dane: {y.index[0].date()} – {y.index[-1].date()} ({len(y)} obs.)")

    fc_parts, timing, vals, grids, best_cfg = [], [], [], [], {}
    for split in SPLITS:
        log(f"=== PODZIAŁ {split}: test od {SPLITS[split].date()} ===")
        log("E1: modele bazowe")
        fc_b, t_b = run_baselines(y, split, log)
        timing += t_b
        if args.skip_grid and (OUT / "best_config.json").exists():
            best_cfg[split] = json.loads((OUT / "best_config.json").read_text())[split]
        else:
            log("E6: przegląd siatki hiperparametrów LSTM")
            gdf, best_cfg[split] = grid_search(y, split, grid, grid_seeds, log)
            grids.append(gdf)
        log(f"  wybrana konfiguracja: {best_cfg[split]}")
        log("E2-E5: warianty LSTM")
        fc_l, t_l, v_l = run_lstm(y, split, best_cfg[split], seeds, mc, log)
        timing += t_l
        vals.append(v_l)
        fc = pd.concat([fc_b, fc_l])
        fc["split"] = split
        fc_parts.append(fc)

    fc_all = pd.concat(fc_parts, ignore_index=True)
    fc_all.to_csv(OUT / "forecasts.csv", index=False)
    (OUT / "best_config.json").write_text(json.dumps(best_cfg, indent=2))
    grid_df = pd.concat(grids) if grids else None
    if grid_df is not None:
        grid_df.to_csv(OUT / "grid_search.csv", index=False)
    val_scores = pd.concat(vals)
    val_scores.to_csv(OUT / "lstm_validation_scores.csv", index=False)

    log("Liczenie metryk i testów DM")
    mt = pd.concat([metrics_table(fc_all[fc_all.split == s], y, s) for s in SPLITS])
    mt.to_csv(OUT / "metrics_per_seed.csv", index=False)
    summ = summarize(mt)
    summ.to_csv(OUT / "metrics_summary.csv", index=False)

    pairs = [(m, "Naive") for m in BASELINES[1:] + ["XGBoost-Δ", "LR-Δ"] + LSTM_NAMES]
    pairs += [(m, "XGBoost") for m in LSTM_NAMES]
    pairs += [(m, "SARIMA") for m in LSTM_NAMES]
    pairs += [("LSTM-Δ", "LSTM-L"), ("XGBoost-Δ", "XGBoost"), ("LSTM-MO", "LSTM-Δ+kal"), ("LSTM-Δ+kal", "LSTM-Δ")]
    dm = pd.concat([dm_table(fc_all[fc_all.split == s], s, pairs) for s in SPLITS])
    dm.to_csv(OUT / "dm_tests.csv", index=False)

    tdf = pd.DataFrame(timing)
    tdf.groupby(["split", "model"])[["train_s", "infer_s"]].mean().reset_index().to_csv(OUT / "timing.csv", index=False)

    report, best_lstm = evaluate_hypotheses(summ, dm, val_scores, best_cfg)
    (OUT / "hypotheses.md").write_text(report, encoding="utf-8")
    (OUT / "best_lstm.json").write_text(json.dumps(best_lstm, indent=2, ensure_ascii=False))
    make_plots(summ, fc_all, grid_df, y, best_lstm)

    # tabela "do sprawozdania"
    key = summ[(summ.period == "całość")].pivot_table(index=["split", "model"], columns="h", values="MAE").round(3)
    key.to_csv(OUT / "table_MAE_by_h.csv")
    print("\n" + key.to_string())
    print("\n" + report)
    log(f"KONIEC. Wyniki w folderze {OUT.resolve()}")


if __name__ == "__main__":
    main()
