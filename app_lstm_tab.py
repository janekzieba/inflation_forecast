"""
Zakładka "LSTM & XAI" aplikacji InflationAI (kamień milowy 3).
Czyta wyniki zapisane przez: experiments.py, xai_lstm.py, lstm_forecast_future.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RES = Path("results")
XAI = RES / "xai"
COLORS = {"Naive": "#7f8c8d", "SARIMA": "#2980b9", "XGBoost": "#c0392b", "XGBoost-Δ": "#e67e22",
          "LR": "#16a085", "RandomForest": "#DAA520", "LSTM-L": "#8e44ad", "LSTM-Δ": "#27ae60",
          "LSTM-Δ+kal": "#2ecc71", "LSTM-MO": "#1abc9c"}


@st.cache_data
def _load():
    d = {}
    for name, path in {
        "summ": RES / "metrics_summary.csv", "dm": RES / "dm_tests.csv", "fc": RES / "forecasts.csv",
        "grid": RES / "grid_search.csv", "timing": RES / "timing.csv", "imp": XAI / "importance.csv",
        "bias": XAI / "bias_by_period.csv", "tp": XAI / "turning_points.csv", "ext": XAI / "extrapolation.csv",
        "top": XAI / "top_errors.csv", "unc": XAI / "uncertainty.csv", "future": Path("lstm_forecast.csv"),
        "expl": Path("lstm_forecast_explanation.csv"),
        "h_met": RES / "final" / "holdout_metrics.csv", "h_dm": RES / "final" / "holdout_dm.csv",
        "h_fc": RES / "final" / "holdout_forecasts.csv", "h_ex": RES / "final" / "exante_12m.csv",
        "h_ig": RES / "final" / "ig_holdout.csv",
    }.items():
        d[name] = pd.read_csv(path) if path.exists() else None
    for k in ("fc",):
        if d[k] is not None:
            d[k]["target_date"] = pd.to_datetime(d[k]["target_date"])
    d["best"] = json.loads((RES / "best_lstm.json").read_text()) if (RES / "best_lstm.json").exists() else {}
    d["cfg"] = json.loads((RES / "best_config.json").read_text()) if (RES / "best_config.json").exists() else {}
    d["hyp"] = (RES / "hypotheses.md").read_text(encoding="utf-8") if (RES / "hypotheses.md").exists() else ""
    d["meta"] = json.loads(Path("lstm_forecast_meta.json").read_text()) if Path("lstm_forecast_meta.json").exists() else {}
    return d


def render(history: pd.DataFrame):
    d = _load()
    if d["summ"] is None:
        st.warning("Brak wyników eksperymentów. Uruchom w terminalu kolejno:\n\n"
                   "`python experiments.py` → `python xai_lstm.py` → `python lstm_forecast_future.py`")
        return

    t1, t2, t3, t4, t5 = st.tabs(["Prognoza LSTM", "Eksperymenty i metryki", "Prognozy testowe", "XAI i analiza błędów",
                                  "Ocena końcowa (nowe dane)"])

    # ------------------------------------------------------------------
    with t1:
        st.markdown("### Prognoza CPI na 12 miesięcy – finalny model LSTM")
        if d["future"] is None:
            st.info("Uruchom `python lstm_forecast_future.py`, aby wygenerować prognozę.")
        else:
            fut = d["future"].copy()
            fut["date"] = pd.to_datetime(fut["date"])
            meta = d["meta"]
            st.caption(f"Wariant: **{meta.get('variant', '?')}**, hiperparametry: `{meta.get('config', {})}`, "
                       f"dane do: {meta.get('data_end', '?')}. Ensemble 5 ziaren losowych.")
            band = st.radio("Przedział 95%", ["konformalny (empiryczny)", "MC-dropout"], horizontal=True)
            st.caption("Przedział konformalny: ± kwantyl 95% |błędów| – " + meta.get("interval_calibration", "") +
                       ". MC-dropout pokazuje tylko niepewność wag sieci i jest zwykle zbyt wąski.")
            lo, hi = ("LSTM_lo_conf", "LSTM_hi_conf") if band.startswith("konf") else ("LSTM_lo", "LSTM_hi")
            h = history.loc["2019":]
            fig = go.Figure()
            fig.add_hrect(y0=101.5, y1=103.5, fillcolor="green", opacity=0.08, line_width=0,
                          annotation_text="cel NBP 2,5% ± 1 p.p.")
            fig.add_trace(go.Scatter(x=h.index, y=h["CPI"], name="CPI (historia)", line=dict(color="black", width=2)))
            fig.add_trace(go.Scatter(x=list(fut.date) + list(fut.date[::-1]), y=list(fut[hi]) + list(fut[lo][::-1]),
                                     fill="toself", fillcolor="rgba(39,174,96,0.2)", line=dict(width=0),
                                     name="LSTM 95%"))
            fig.add_trace(go.Scatter(x=fut.date, y=fut.LSTM, name="LSTM", line=dict(color="#27ae60", width=3)))
            for f, col, name, c in [("sarima_10y_forecast.csv", "CPI_forecast", "SARIMA", "#2980b9"),
                                    ("ml_10y_forecast_combined.csv", "CPI_forecast_XGB", "XGBoost", "#c0392b")]:
                if Path(f).exists():
                    o = pd.read_csv(f, parse_dates=["date"]).head(12)
                    fig.add_trace(go.Scatter(x=o.date, y=o[col], name=name, line=dict(color=c, dash="dash")))
            fig.update_layout(height=480, template="plotly_white", yaxis_title="CPI (r/r, indeks)")
            st.plotly_chart(fig, width="stretch")
            show = fut[["date", "LSTM", lo, hi, "LSTM_seed_std"]].copy()
            show["inflacja [%]"] = show["LSTM"] - 100
            show["date"] = show["date"].dt.strftime("%Y-%m")
            st.dataframe(show.round(2), hide_index=True, width="stretch")

            if d["expl"] is not None:
                st.markdown("#### Dlaczego taka prognoza? (Integrated Gradients)")
                e = d["expl"].copy()
                e["input_month"] = pd.to_datetime(e["input_month"]).dt.strftime("%Y-%m")
                colors = ["#c0392b" if v > 0 else "#2980b9" for v in e.contribution_pp]
                fig = go.Figure(go.Bar(x=e.input_month, y=e.contribution_pp, marker_color=colors,
                                       customdata=np.c_[e.CPI, e.delta],
                                       hovertemplate="%{x}<br>wkład: %{y:.3f} p.p.<br>CPI: %{customdata[0]:.1f}"
                                                     "<br>Δ m/m: %{customdata[1]:+.1f}<extra></extra>"))
                fig.update_layout(height=320, template="plotly_white", yaxis_title="wkład w prognozę zmiany [p.p.]",
                                  title="Wkład każdego miesiąca wejściowego w prognozę na następny miesiąc")
                st.plotly_chart(fig, width="stretch")
                top = e.reindex(e.contribution_pp.abs().sort_values(ascending=False).index).head(3)
                txt = "; ".join(f"{r.input_month} (Δ = {r.delta:+.1f}) → {r.contribution_pp:+.3f} p.p."
                                for r in top.itertuples())
                st.markdown(f"Największy wpływ na prognozę mają miesiące: {txt}. Czerwony słupek podnosi, "
                            f"niebieski obniża prognozowaną zmianę CPI względem przeciętnej zmiany z danych "
                            f"treningowych (linia bazowa IG).")

    # ------------------------------------------------------------------
    with t2:
        summ = d["summ"]
        c1, c2, c3 = st.columns(3)
        split = c1.selectbox("Podział danych", ["B", "A"], format_func=lambda s: {
            "A": "A – trening 2001-2014, test 2015-", "B": "B – trening 2001-2022, test 2023- (zmiana reżimu)"}[s])
        period = c2.selectbox("Podokres", summ.period.unique().tolist())
        hsel = c3.selectbox("Horyzont h [mies.]", sorted(summ.h.unique()))
        st.caption(f"Najlepszy wariant LSTM (wybrany na walidacji): **{d['best'].get(split)}**, "
                   f"hiperparametry: `{d['cfg'].get(split)}`")
        s = summ[(summ.split == split) & (summ.period == period)]
        tab = s[s.h == hsel][["model", "MAE", "MAE_std", "RMSE", "MAPE", "MASE", "RelMAE", "DA", "coverage95",
                              "coverage95_conf", "n_seeds", "n"]].sort_values("MAE")
        st.dataframe(tab.round(3), hide_index=True, width="stretch")
        st.caption("RelMAE = MAE / MAE(Naive) na tych samych punktach (< 1 = lepiej niż Naive). "
                   "DA = trafność kierunku [%]. coverage95 = pokrycie 95% (MC-dropout / SARIMA); "
                   "coverage95_conf = pokrycie przedziałów konformalnych.")
        fig = go.Figure()
        for m in ["Naive", "SARIMA", "XGBoost", "XGBoost-Δ", "LR", "LSTM-L", "LSTM-Δ", "LSTM-Δ+kal", "LSTM-MO"]:
            r = s[s.model == m].sort_values("h")
            if len(r):
                fig.add_trace(go.Scatter(x=r.h, y=r.MAE, name=m, mode="lines+markers",
                                         error_y=dict(type="data", array=r.MAE_std, visible=True),
                                         line=dict(color=COLORS.get(m), dash="solid" if m.startswith("LSTM") else "dash")))
        fig.update_layout(height=420, template="plotly_white", xaxis_title="h [mies.]", yaxis_title="MAE [p.p.]",
                          title="MAE w funkcji horyzontu (słupki błędów = odch. std po ziarnach)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Test Diebolda-Mariano")
        dm = d["dm"]
        dsel = dm[(dm.split == split) & (dm.period == period) & (dm.h == hsel)].copy()
        dsel["istotne (α=0,05)"] = np.where(dsel.p_model_better < 0.05, "model lepszy",
                                            np.where(dsel.p_two_sided < 0.05, "różnica istotna", "brak"))
        st.dataframe(dsel[["model", "reference", "n", "MAE_model", "MAE_ref", "DM", "p_two_sided",
                           "p_model_better", "istotne (α=0,05)"]].round(4), hide_index=True, width="stretch")
        with st.expander("Weryfikacja hipotez H1-H4 i kryteriów sukcesu"):
            st.markdown(d["hyp"])
        if d["grid"] is not None:
            with st.expander("E6: przegląd siatki hiperparametrów (walidacja krocząca)"):
                g = d["grid"][d["grid"].split == split]
                agg = g.groupby(["window", "hidden", "dropout", "lr"])["val_MAE"].agg(["mean", "std"]).reset_index()
                st.dataframe(agg.sort_values("mean").round(4), hide_index=True, width="stretch")
        if d["timing"] is not None:
            with st.expander("Koszt obliczeniowy (czas treningu i inferencji)"):
                st.dataframe(d["timing"][d["timing"].split == split].round(3), hide_index=True, width="stretch")

    # ------------------------------------------------------------------
    with t3:
        fc = d["fc"]
        c1, c2 = st.columns(2)
        split3 = c1.selectbox("Podział", ["B", "A"], key="split3")
        h3 = c2.slider("Horyzont h", 1, 12, 1)
        models = sorted(fc.model.unique())
        default = [m for m in ["Naive", "XGBoost", "LSTM-L", d["best"].get(split3)] if m in models]
        chosen = st.multiselect("Modele", models, default=list(dict.fromkeys(default)))
        g = fc[(fc.split == split3) & (fc.h == h3)]
        avg = g.groupby(["model", "target_date"])[["y_true", "y_pred", "lo", "hi", "lo_conf", "hi_conf"]].mean().reset_index()
        fig = go.Figure()
        hist = history.loc["2012":]
        fig.add_trace(go.Scatter(x=hist.index, y=hist.CPI, name="CPI rzeczywiste", line=dict(color="black", width=2)))
        for m in chosen:
            a = avg[avg.model == m]
            if m.startswith("LSTM") and a.lo_conf.notna().all():
                fig.add_trace(go.Scatter(x=list(a.target_date) + list(a.target_date[::-1]),
                                         y=list(a.hi_conf) + list(a.lo_conf[::-1]), fill="toself", opacity=0.15,
                                         line=dict(width=0), name=f"{m} 95% (konf.)", showlegend=True))
            fig.add_trace(go.Scatter(x=a.target_date, y=a.y_pred, name=m, line=dict(color=COLORS.get(m))))
        fig.update_layout(height=500, template="plotly_white", title=f"Prognozy {h3}-krokowe (rolling origin)",
                          yaxis_title="CPI")
        st.plotly_chart(fig, width="stretch")
        st.caption("Dla modeli LSTM prognoza to średnia z ensemble ziaren losowych. Każdy punkt to prognoza "
                   f"wykonana {h3} mies. wcześniej, na podstawie danych dostępnych w tamtym momencie.")

    # ------------------------------------------------------------------
    with t4:
        if d["imp"] is None:
            st.info("Uruchom `python xai_lstm.py`.")
            return
        split4 = st.selectbox("Podział", ["A", "B"], key="split4")
        imp = d["imp"]
        c1, c2 = st.columns(2)
        sub = imp[(imp.split == split4) & (imp.method == "IG |atrybucja|")]
        fig = go.Figure()
        for m, gg in sub.groupby("model"):
            fig.add_trace(go.Scatter(x=gg.lag.astype(int), y=gg.importance, name=m, mode="lines+markers",
                                     line=dict(color=COLORS.get(m))))
        fig.update_layout(height=380, template="plotly_white", title="LSTM: ważność opóźnień (Integrated Gradients)",
                          xaxis_title="opóźnienie [mies.]", yaxis_title="średni |wkład| [p.p.]")
        c1.plotly_chart(fig, width="stretch")
        sh = imp[(imp.split == split4) & (imp.model == "XGBoost")].sort_values("importance")
        fig = go.Figure(go.Bar(x=sh.importance, y=sh.lag, orientation="h", marker_color="#c0392b"))
        fig.update_layout(height=380, template="plotly_white", title="XGBoost: ważność cech (SHAP)",
                          xaxis_title="średni |SHAP| [p.p.]")
        c2.plotly_chart(fig, width="stretch")

        occ = imp[(imp.split == split4) & (imp.method == "Okluzja ΔMAE")]
        if len(occ):
            fig = go.Figure()
            for m, gg in occ.groupby("model"):
                fig.add_trace(go.Bar(x=gg.lag.astype(int), y=gg.importance, name=m))
            fig.update_layout(height=320, template="plotly_white", barmode="group",
                              title="Okluzja: wzrost MAE (h=1) po zastąpieniu danego miesiąca średnią",
                              xaxis_title="opóźnienie [mies.]", yaxis_title="ΔMAE [p.p.]")
            st.plotly_chart(fig, width="stretch")

        best = d["best"].get(split4)
        hp = XAI / f"ig_heatmap_{split4}_{best}.csv"
        if hp.exists():
            heat = pd.read_csv(hp, index_col=0, parse_dates=True)
            lim = float(np.nanpercentile(np.abs(heat.values), 98))
            fig = go.Figure(go.Heatmap(z=heat.values.T, x=heat.index, y=list(range(1, heat.shape[1] + 1)),
                                       colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=dict(title="p.p.")))
            fig.update_layout(height=380, template="plotly_white", yaxis_title="opóźnienie",
                              title=f"Integrated Gradients w czasie – {best} (czerwony: podnosi prognozę)")
            st.plotly_chart(fig, width="stretch")

        if hp.exists():
            st.markdown("#### Wyjaśnienie pojedynczej prognozy")
            heat = pd.read_csv(hp, index_col=0, parse_dates=True)
            dsel = st.select_slider("Miesiąc prognozowany (prognoza 1-miesięczna)", options=list(heat.index),
                                    value=heat.index[-1], format_func=lambda d: d.strftime("%Y-%m"), key="loc_xai")
            row = heat.loc[dsel]
            months = pd.date_range(end=dsel - pd.DateOffset(months=1), periods=len(row), freq="MS")[::-1]
            fig = go.Figure(go.Bar(x=[m.strftime("%Y-%m") for m in months], y=row.values,
                                   marker_color=["#c0392b" if v > 0 else "#2980b9" for v in row.values]))
            fc_ = d["fc"]
            g = fc_[(fc_.split == split4) & (fc_.model == best) & (fc_.h == 1) & (fc_.target_date == dsel)]
            ttl = (f"Prognoza na {dsel:%Y-%m}: {g.y_pred.mean():.2f} (rzeczywiste {g.y_true.mean():.1f}) – "
                   f"wkład miesięcy wejściowych" if len(g) else "Wkład miesięcy wejściowych")
            fig.update_layout(height=320, template="plotly_white", title=ttl, yaxis_title="wkład [p.p.]",
                              xaxis_title="miesiąc wejściowy (od najnowszego)")
            st.plotly_chart(fig, width="stretch")

        st.markdown("### Gdzie i dlaczego modele się mylą?")
        b = d["bias"]
        st.markdown("**Obciążenie (średni błąd ze znakiem, h=1; > 0 = model zaniża, < 0 = zawyża)**")
        st.dataframe(b[(b.split == split4) & (b.h == 1)].pivot_table(index="model", columns="period", values="bias")
                     .round(3), width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown("**Punkty zwrotne** (zmiana znaku 3-mies. dynamiki CPI)")
        c1.dataframe(d["tp"][(d["tp"].split == split4)].round(2), hide_index=True, width="stretch")
        c2.markdown("**Ekstrapolacja poza maksimum treningu** (Podział A)")
        c2.dataframe(d["ext"].round(3), hide_index=True, width="stretch")
        st.markdown("**Największe błędy najlepszego LSTM**")
        st.dataframe(d["top"][d["top"].split == split4][["model", "h", "origin", "target_date", "y_true", "y_pred",
                                                          "err"]].round(3), hide_index=True, width="stretch")
        st.markdown("**Niepewność: czy model wie, kiedy nie wie?**")
        st.dataframe(d["unc"][d["unc"].split == split4].round(3), hide_index=True, width="stretch")
        st.caption("MC-dropout mierzy tylko niepewność wag modelu, więc zwykle daje zbyt wąskie przedziały. "
                   "Przedziały konformalne są kalibrowane na błędach z walidacji kroczącej.")

    # ------------------------------------------------------------------
    with t5:
        st.markdown("### Ocena końcowa na danych, których żaden model nie widział (11.2025 – 10.2026)")
        if d["h_met"] is None:
            st.info("Uruchom `python final_evaluation.py`.")
            return
        st.caption("Modele uczone wyłącznie na danych do 10.2025 i zamrożone. Nowe odczyty CPI (GUS) trafiają "
                   "co miesiąc na wejście, parametry się nie zmieniają. Od 01.2026 GUS liczy CPI wg COICOP 2018.")
        ex = d["h_ex"].copy()
        ex["target_date"] = pd.to_datetime(ex["target_date"])
        fig = go.Figure()
        hist = history.loc["2024-06":]
        fig.add_trace(go.Scatter(x=hist.index, y=hist.CPI, name="CPI (dane uczące)", line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=ex.target_date, y=ex["rzeczywiste"], name="CPI rzeczywiste (nowe dane)",
                                 line=dict(color="black", width=3, dash="dot"), mode="lines+markers"))
        fig.add_trace(go.Scatter(x=list(ex.target_date) + list(ex.target_date[::-1]),
                                 y=list(ex.LSTM_hi_conf) + list(ex.LSTM_lo_conf[::-1]), fill="toself",
                                 fillcolor="rgba(39,174,96,0.12)", line=dict(width=0), name="LSTM-Δ 95%"))
        for m in ["Naive", "SARIMA", "XGBoost", "LR-Δ", "LSTM-Δ"]:
            if m in ex:
                fig.add_trace(go.Scatter(x=ex.target_date, y=ex[m], name=f"{m} (z 10.2025)", line=dict(
                    color=COLORS.get(m, "#16a085"), dash="solid" if m == "LSTM-Δ" else "dash")))
        fig.update_layout(height=430, template="plotly_white", yaxis=dict(range=[97, 109], title="CPI"),
                          title="Ocena ex ante: prognoza na 12 miesięcy wykonana w październiku 2025")
        st.plotly_chart(fig, width="stretch")
        mae_ex = {m: np.mean(np.abs(ex["rzeczywiste"] - ex[m])) for m in ["Naive", "SARIMA", "LR", "LR-Δ", "XGBoost",
                                                                           "XGBoost-Δ", "LSTM-L", "LSTM-MO", "LSTM-Δ"] if m in ex}
        st.dataframe(pd.DataFrame({"MAE ex ante (12 mies.)": mae_ex}).sort_values("MAE ex ante (12 mies.)").round(3),
                     width="stretch")

        met = d["h_met"]
        st.markdown("#### Prognozy kroczące na nowych danych")
        hsel = st.radio("Horyzont", [1, 3, 6], horizontal=True, key="h_hold")
        st.dataframe(met[met.h == hsel][["model", "n", "MAE", "RMSE", "RelMAE", "DA", "coverage95_conf",
                                         "MAE_std_seeds"]].sort_values("MAE").round(3), hide_index=True, width="stretch")
        dm = d["h_dm"]
        st.markdown("**Test Diebolda-Mariano (nowe dane)**")
        st.dataframe(dm[(dm.h == hsel) & dm.model.str.startswith("LSTM-Δ")].round(3), hide_index=True, width="stretch")
        if d["h_ig"] is not None:
            ig = d["h_ig"].set_index("target_date")
            fig = go.Figure(go.Heatmap(z=ig.values.T, x=ig.index, y=list(range(1, ig.shape[1] + 1)),
                                       colorscale="RdBu_r", zmid=0, colorbar=dict(title="p.p.")))
            fig.update_layout(height=330, template="plotly_white", yaxis_title="opóźnienie",
                              title="XAI na nowych danych: Integrated Gradients dla każdej prognozy 1-miesięcznej")
            st.plotly_chart(fig, width="stretch")
