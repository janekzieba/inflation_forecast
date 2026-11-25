import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.io import savemat
from fpdf import FPDF
import io
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, norm

# ==========================================
# 1. KONFIGURACJA STRONY I STYL (CSS)
# ==========================================
st.set_page_config(page_title="InflationAI - System Predykcji Inflacji w Polsce", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    /* Styl dla naszych własnych kart */
    .custom-card {
        background-color: var(--secondary-background-color) !important;
        padding: 20px; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    /* Wymuszenie kolorów tekstu */
    .custom-card h1, .custom-card h2, .custom-card h3, .custom-card h4, .custom-card p, .custom-card li, .custom-card span, .custom-card div {
        color: var(--text-color);
    }
    .label-text { opacity: 0.7; font-size: 0.9em; text-transform: uppercase; font-weight: 600; margin: 0; }

    /* Przyciski */
    .stButton button { background-color: #2c3e50 !important; color: white !important; border: none; }
    .stButton button:hover { background-color: #34495e !important; }

    /* Baner powrotu do celu */
    .target-banner {
        background-color: var(--secondary-background-color) !important;
        padding: 15px; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;
        border: 2px solid #27ae60;
        text-align: center;
    }
    .target-banner h3, .target-banner p { color: var(--text-color); }

    /* Listy */
    ul, ol { padding-left: 20px; }
    li { margin-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. FUNKCJE POMOCNICZE
# ==========================================
@st.cache_data
def load_data_app():
    try:
        hist = pd.read_csv("cpi_data.csv", parse_dates=['date']).set_index('date')
        sarima = pd.read_csv("sarima_10y_forecast.csv", parse_dates=['date']).set_index('date')
        ml = pd.read_csv("ml_10y_forecast_combined.csv", parse_dates=['date']).set_index('date')
        naive = pd.read_csv("naive_10y_forecast.csv", parse_dates=['date']).set_index('date')
        val_data = pd.read_csv("validation_data.csv", parse_dates=['date']).set_index('date')
        # --- LSTM (kamienie milowe 2-3): prognoza finalnego modelu + prognozy testowe do ewaluacji ---
        lstm = None
        try:
            lstm = pd.read_csv("lstm_forecast.csv", parse_dates=['date']).set_index('date')
        except FileNotFoundError:
            pass
        try:
            fc = pd.read_csv("results/forecasts.csv", parse_dates=['target_date'])
            sel = fc[(fc.split == "A") & (fc.model == "LSTM-Δ") & (fc.h == 1)]
            lstm_val = sel.groupby('target_date')['y_pred'].mean()   # ensemble 5 ziaren, prognoza 1-mies.
            val_data['LSTM'] = lstm_val.reindex(val_data.index)
        except FileNotFoundError:
            pass
        return hist, sarima, ml, naive, val_data, lstm
    except FileNotFoundError:
        st.error("Brak plików! Uruchom najpierw 'run_all.py' oraz 'generate_validation.py'!")
        st.stop()


def get_inflation_status(val):
    rate = val - 100
    if rate < 0: return "DEFLACJA", "#3498db"
    if rate < 1.5: return "BARDZO NISKA", "#f39c12"
    if rate <= 3.5: return "W CELU NBP", "#27ae60"
    if rate <= 8.0: return "WYSOKA", "#e67e22"
    return "KRYZYSOWA", "#e74c3c"


def calculate_return_date(forecast_series):
    # Cel NBP: 2.5% +/- 1 p.p. => Wskaźnik 101.5 do 103.5
    in_target = forecast_series[(forecast_series >= 101.5) & (forecast_series <= 103.5)]
    if not in_target.empty:
        return in_target.index[0]
    return None


def generate_pdf_report(date_str, value, status, model_name, real_val_1000):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    def clean(text):
        replacements = {'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
                        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z',
                        'Δ': 'Delta', '–': '-'}
        for k, v in replacements.items(): text = str(text).replace(k, v)
        return text

    pdf.cell(200, 10, txt=clean("Raport Predykcji Inflacji CPI"), ln=1, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=clean(f"Miesiac prognozy: {date_str}"), ln=1, align='L')
    pdf.cell(200, 10, txt=clean(f"Wybrany model: {model_name}"), ln=1, align='L')
    pdf.cell(200, 10, txt=clean(f"Przewidywany wskaznik CPI: {value:.2f}"), ln=1, align='L')
    pdf.cell(200, 10, txt=clean(f"Status inflacji: {status}"), ln=1, align='L')
    pdf.ln(10)
    pdf.cell(200, 10, txt=clean(f"Sila nabywcza 1000 PLN: {real_val_1000:.2f} PLN"), ln=1, align='L')
    pdf.ln(20)
    pdf.cell(200, 10, txt=clean("Wygenerowano przez System InflationAI"), ln=1, align='C')
    return pdf.output(dest='S').encode('latin-1', errors='ignore')


def display_kpi(title, val, delta, d_col, status, s_col):
    st.markdown(f"""
    <div class="custom-card" style="border-left: 8px solid {s_col};">
        <p class="label-text">{title}</p>
        <h2 style="margin:5px 0; font-size:2.5em; font-weight:700;">{val}</h2>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:{d_col} !important; font-weight:bold; background:{d_col}20; padding:4px 8px; border-radius:5px;">{delta}</span>
            <span style="color:{s_col} !important; font-weight:bold;">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def calculate_metrics(df_segment):
    results = []
    models = ['XGBoost', 'RandomForest', 'SARIMA', 'Naive'] + (['LSTM'] if 'LSTM' in df_segment else [])
    df_segment = df_segment.dropna()
    for m in models:
        y_true, y_pred = df_segment['Actual'], df_segment[m]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        results.append({"Model": m, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})
    return pd.DataFrame(results).sort_values("MAE")


# ==========================================
# 3. ŁADOWANIE DANYCH
# ==========================================
history, sarima, ml, naive, val_data, lstm = load_data_app()
HAS_LSTM = lstm is not None

# ==========================================
# 4. INTERFEJS - SIDEBAR (GLOBALNY)
# ==========================================
with st.sidebar:
    st.title("Panel Sterowania")
    st.markdown("---")

    # Sekcja A: Ustawienia Prognozy
    st.markdown("### Ustawienia Wykresu")
    horizon = st.slider("Horyzont (m-ce)", 1, 120, 24)

    st.markdown("### Widoczność")
    show_lstm = st.checkbox("LSTM-Δ (moduł AI)", True, disabled=not HAS_LSTM)
    show_xgb = st.checkbox("XGBoost (ML)", True)
    show_rf = st.checkbox("Random Forest", True)
    show_sarima = st.checkbox("SARIMA", True)
    show_naive = st.checkbox("Naive", False)

    st.markdown("---")

    # Sekcja B: Symulacja
    st.markdown("### Symulacja (What-If)")
    if "shock" not in st.session_state: st.session_state.shock = 0.0

    def reset_shock():
        st.session_state.shock = 0.0

    shock = st.slider("Szok (p.p.)", -5.0, 5.0, step=0.1, key="shock")
    st.button("↺ Resetuj Szok", on_click=reset_shock)
    if shock != 0: st.warning(f" Symulacja aktywna: {shock:+.1f} p.p.")

    st.markdown("---")

    # Sekcja C: Ustawienia KPI
    st.markdown("### Ustawienia KPI")
    target_date = st.selectbox("Miesiąc KPI:", ml.index, format_func=lambda x: x.strftime('%Y-%m'), index=11)
    kpi_models = (["LSTM-Δ (Rekomendowany, AI)"] if HAS_LSTM else []) + ["XGBoost", "Random Forest", "SARIMA", "Naive"]
    target_model = st.selectbox("Model KPI:", kpi_models)
    if "LSTM" in target_model:
        st.caption("LSTM prognozuje 12 miesięcy naprzód (dalej prognoza rekurencyjna traci sens).")

    st.caption("System InflationAI")

# ==========================================
# 5. STRUKTURA GŁÓWNA (TABS)
# ==========================================
st.title("InflationAI - System Predykcji Inflacji CPI w Polsce")
st.markdown(f"Stan danych na: **{history.index[-1].strftime('%Y-%m')}**")

tab1, tab2, tab3 = st.tabs(["Dashboard (Prognoza)", "Porównanie Modeli (Ewaluacja)", "LSTM & XAI (projekt AI)"])

# ==================================================
# TAB 1: DASHBOARD (PROGNOZA + KALKULATOR + RAPORT)
# ==================================================
with tab1:
    # --- 1. KPI CARDS I DANE ---
    last, prev = history['CPI'].iloc[-1], history['CPI'].iloc[-2]
    model_key = target_model.split()[0]

    # Pobranie pełnej serii prognozy dla wybranego modelu
    if "LSTM" in target_model:
        forecast_series = lstm['LSTM']
    elif "XGBoost" in target_model:
        forecast_series = ml['CPI_forecast_XGB']
    elif "Random" in target_model:
        forecast_series = ml['CPI_forecast_RF']
    elif "SARIMA" in target_model:
        forecast_series = sarima['CPI_forecast']
    else:
        forecast_series = naive['Naive_Seasonal']

    # Aplikacja szoku do całej serii
    forecast_series = forecast_series + shock

    # Pobranie punktu dla KPI
    try:
        pred = forecast_series.loc[target_date]
    except:
        pred = forecast_series.iloc[-1] if "LSTM" in target_model else last
        if "LSTM" in target_model:
            st.info(f"LSTM prognozuje do {forecast_series.index[-1].strftime('%Y-%m')} – KPI pokazuje ostatni miesiąc prognozy.")

    d_now = last - prev
    d_col = "#27ae60" if d_now <= 0 else "#e74c3c"
    stat_now, _ = get_inflation_status(last)

    d_fut = pred - last
    d_fut_col = "#27ae60" if d_fut <= 0 else "#e74c3c"
    stat_fut, col_fut = get_inflation_status(pred)
    date_str = target_date.strftime('%Y-%m')

    c1, c2, c3 = st.columns(3)
    with c1:
        display_kpi("Aktualny Odczyt", f"{last:.1f}", f"{d_now:+.1f} m/m", d_col, stat_now,
                    "#27ae60")  # Zielony jak w celu
    with c2:
        display_kpi("Cel NBP", "102.5", "Stały", "#95a5a6", "Optymalny", "#2c3e50")

    shock_txt = f" (Szok {shock:+.1f})" if shock != 0 else ""
    with c3:
        display_kpi(f"Prognoza ({date_str}){shock_txt}", f"{pred:.1f}", f"{d_fut:+.1f} vs teraz", d_fut_col, stat_fut,
                    col_fut)

    # --- 2. BANER "POWRÓT DO CELU" (TERAZ WYSRODKOWANY!) ---
    return_date = calculate_return_date(forecast_series)

    if return_date:
        ret_str = return_date.strftime('%Y-%m')
        quarter = (return_date.month - 1) // 3 + 1
        # Formatowanie HTML z kolorem zielonym i pogrubieniem
        date_html = f"<span style='color: #27ae60; font-weight: bold;'>{ret_str} (Q{quarter} {return_date.year})</span>"
        banner_msg = f" Model sugeruje powrót inflacji do celu NBP (2.5% ± 1p.p.) w: {date_html}"
        banner_border = "#27ae60"  # Zielony
    else:
        banner_msg = " W prognozowanym horyzoncie (10 lat) model nie przewiduje trwałego powrotu do celu NBP."
        banner_border = "#e74c3c"  # Czerwony

    # Wypisujemy baner w głównym bloku (POZA KOLUMNAMI)
    st.markdown(f"""
    <div class="target-banner" style="border-color: {banner_border}; margin-top: 20px;">
        <h3 style="margin:0;">{banner_msg}</h3>
        <p style="margin:0; opacity:0.7; font-size:0.9em;">Analiza dla modelu: {model_key} {shock_txt}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 3. WYKRES PROGNOZY (Z OPISEM CELU) ---
    st.subheader("Wizualizacja Szeregów Czasowych")
    fig = go.Figure()

    start_p = pd.to_datetime("2018-01-01")
    max_p = history.index[-1] + pd.DateOffset(months=horizon)
    hist_p = history.loc[start_p:]

    fig.add_trace(go.Scatter(x=hist_p.index, y=hist_p['CPI'], name='Historia', line=dict(color='black', width=3)))

    if show_lstm and HAS_LSTM:
        l_p = lstm[:max_p]
        fig.add_trace(go.Scatter(x=l_p.index.tolist()[::-1] + l_p.index.tolist(),
                                 y=l_p['LSTM_lo_conf'].tolist()[::-1] + l_p['LSTM_hi_conf'].tolist(),
                                 fill='toself', fillcolor='rgba(39,174,96,0.10)', line=dict(color='rgba(0,0,0,0)'),
                                 name='LSTM-Δ 95% (konformalny)', hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=l_p.index, y=l_p['LSTM'], name='LSTM-Δ (AI)',
                                 line=dict(color='#27ae60', width=3)))
    if show_xgb: fig.add_trace(
        go.Scatter(x=ml[:max_p].index, y=ml[:max_p]['CPI_forecast_XGB'], name='XGBoost', line=dict(color='#e74c3c')))
    if show_rf: fig.add_trace(go.Scatter(x=ml[:max_p].index, y=ml[:max_p]['CPI_forecast_RF'], name='Random Forest',
                                         line=dict(color='#f1c40f')))
    if show_sarima:
        fig.add_trace(go.Scatter(x=sarima[:max_p].index, y=sarima[:max_p]['CPI_forecast'], name='SARIMA',
                                 line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=sarima[:max_p].index.tolist()[::-1] + sarima[:max_p].index.tolist(),
                                 y=sarima[:max_p]['lower_95'].tolist()[::-1] + sarima[:max_p]['upper_95'].tolist(),
                                 fill='toself', fillcolor='rgba(52,152,219,0.1)', line=dict(color='rgba(0,0,0,0)'),
                                 showlegend=False, hoverinfo='skip'))
    if show_naive: fig.add_trace(go.Scatter(x=naive[:max_p].index, y=naive[:max_p]['Naive_Seasonal'], name='Naive',
                                            line=dict(color='purple', dash='dash')))

    if shock != 0:
        sim_plot = forecast_series[:max_p]
        fig.add_trace(go.Scatter(x=sim_plot.index, y=sim_plot, name=f'Symulacja ({shock:+.1f})',
                                 line=dict(color='black', dash='dot', width=2)))

    fig.add_hrect(
        y0=101.5, y1=103.5,
        line_width=0, fillcolor="green", opacity=0.1,
        annotation_text="Cel inflacyjny NBP (2.5% ± 1 p.p.)",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="green"
    )

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="Rok", yaxis_title="CPI",
                      template="plotly_white", xaxis=dict(range=[start_p, max_p]))
    st.plotly_chart(fig, width="stretch")

    try:
        img = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button("Pobierz Wykres (PNG)", img, "wykres_hq.png", "image/png")
    except:
        st.warning("Zainstaluj kaleido aby pobrać PNG")

    st.markdown("---")

    # --- 4. INTERPRETACJA + KALKULATOR ---
    col_ai, col_calc = st.columns(2)

    with col_ai:
        st.subheader("Interpretacja modelu")
        if "LSTM" in target_model:
            model_key = "LSTM-Δ"
        elif "Random" in target_model:
            model_key = "Random Forest"
        elif "XGBoost" in target_model:
            model_key = "XGBoost"
        elif "SARIMA" in target_model:
            model_key = "SARIMA"
        else:
            model_key = "Naive"

        # Opis interpretacyjny
        if "LSTM" in target_model:
            try:
                ex = pd.read_csv("lstm_forecast_explanation.csv", parse_dates=['input_month'])
                top = ex.reindex(ex['contribution_pp'].abs().sort_values(ascending=False).index).head(3)
                items = "".join(f"<li><b>{r.input_month.strftime('%Y-%m')}</b> (zmiana CPI {r.delta:+.1f}): "
                                f"wkład {r.contribution_pp:+.3f} p.p.</li>" for r in top.itertuples())
            except FileNotFoundError:
                items = "<li>Brak pliku z wyjaśnieniem – uruchom lstm_forecast_future.py</li>"
            desc = f"""
            <p>Sieć <b>LSTM-Δ</b> prognozuje <b>miesięczną zmianę</b> CPI na podstawie 12 ostatnich zmian.
            Wyjaśnienie metodą <i>Integrated Gradients</i> – miesiące, które najmocniej wpłynęły na prognozę:</p>
            <ul>{items}</ul>
            <p style="opacity:0.7;font-size:0.9em;">Szczegóły: zakładka „LSTM & XAI”.</p>
            """
        elif "XGBoost" in target_model or "Random" in target_model:
            desc = f"""
            <p>Dla modelu <b>{model_key}</b> kluczowa jest historia:</p>
            <ul>
                <li>
                    <b>Najważniejszy czynnik:</b> Inflacja z poprzedniego miesiąca (<code>lag_1</code>).
                    Model zakłada dużą inercję cen.
                </li>
                <li>
                    <b>Sezonowość:</b> W okresach kryzysowych ma mniejsze znaczenie.
                </li>
            </ul>
            """
        elif "SARIMA" in target_model:
            desc = """
            <p>Model <b>SARIMA</b> opiera się na matematyce szeregów:</p>
            <ul>
                <li><b>Autoregresja:</b> Wynik zależy od ostatnich 2 miesięcy.</li>
                <li><b>Powrót do średniej:</b> Model dąży do stabilizacji wskaźnika w długim terminie.</li>
            </ul>
            """
        else:
            desc = """
            <p>Model <b>Naiwny</b> kopiuje historię:</p>
            <ul>
                <li>Zakłada, że przyszłość będzie dokładnym powtórzeniem ostatnich 12 miesięcy.</li>
            </ul>
            """
        st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0;">Dlaczego taka prognoza?</h4>{desc}</div>""",
                    unsafe_allow_html=True)

    with col_calc:
        st.subheader("Kalkulator Siły Nabywczej")
        amount = st.number_input("Wpisz kwotę (PLN):", value=1000.0, step=100.0, min_value=0.0)
        ratio = last / pred if pred > 0 else 1
        real_val = amount * ratio
        loss = amount - real_val
        l_col = "#e74c3c" if loss > 0 else "#27ae60"
        st.markdown(f"""
        <div class="custom-card">
            <p class="label-text">Wartość w {date_str}:</p>
            <h2 style="margin:10px 0; font-size:2.5em; color:#2c3e50 !important;">{real_val:.2f} PLN</h2>
            <div style="background:{l_col}20; padding:10px; border-radius:5px; border-left:5px solid {l_col};">
                <strong style="color:{l_col} !important;">Zmiana: {'-' if loss > 0 else '+'}{abs(loss):.2f} PLN</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    # --- 5. EKSPORT RAPORTÓW ---
    st.markdown("---")
    st.subheader("Eksport Raportów")
    c1, c2, c3, spacer = st.columns([1, 1, 1, 5])

    csv_buf = io.StringIO()
    pd.concat([ml, sarima['CPI_forecast']], axis=1).to_csv(csv_buf)

    with c1:
        st.download_button("Pobierz CSV", csv_buf.getvalue(), "dane.csv", "text/csv")
    with c2:
        pdf_val_1000 = 1000 * (last / pred)
        pdf_b = generate_pdf_report(date_str, pred, stat_fut, model_key, pdf_val_1000)
        st.download_button("Raport PDF", pdf_b, f"raport_{date_str}.pdf", "application/pdf")
    with c3:
        mat_buf = io.BytesIO()
        savemat(mat_buf, {'data': ml.to_dict("list")})
        st.download_button("MATLAB (.mat)", mat_buf.getvalue(), "dane.mat", "application/octet-stream")

# ==================================================
# TAB 2: EWALUACJA I DIAGNOSTYKA (NAPRAWIONA PACF)
# ==================================================
with tab2:
    st.header("Ewaluacja i Porównanie Modeli")
    st.info("Analiza skuteczności modeli na danych historycznych (Backtesting).")

    col_sel, col_chart = st.columns([1, 3])

    with col_sel:
        st.markdown("### Okres testowy")
        period = st.radio("Wybierz zakres:", ["Cały (2015-2025)", "Kryzys (2020-2025)", "Ostatnie 2 lata"], index=2)
        if "2015" in period:
            start_v = "2015-01-01"
        elif "2020" in period:
            start_v = "2020-01-01"
        else:
            start_v = "2023-01-01"

        subset = val_data.loc[start_v:]
        metrics_df = calculate_metrics(subset)

        st.markdown("---")
        st.markdown("#### Ranking (MAE)")
        st.dataframe(
            metrics_df[['Model', 'MAE']].style
            .highlight_min(subset=['MAE'], axis=0, color='#d4edda')
            .format("{:.4f}", subset=['MAE']),
            hide_index=True,
            width="stretch"
        )

    with col_chart:
        st.subheader(f"Błędy modeli (MAE/RMSE) od {start_v}")
        fig_bar = px.bar(metrics_df, x='Model', y=['MAE', 'RMSE'], barmode='group',
                         color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig_bar, width="stretch")

        st.subheader("Wykres Rzeczywistość vs Prognozy")
        fig_l = go.Figure()
        fig_l.add_trace(
            go.Scatter(x=subset.index, y=subset['Actual'], name='Rzeczywistość', line=dict(color='black', width=3)))
        fig_l.add_trace(
            go.Scatter(x=subset.index, y=subset['XGBoost'], name='XGBoost', line=dict(color='#e74c3c', dash='dot')))
        fig_l.add_trace(go.Scatter(x=subset.index, y=subset['RandomForest'], name='RandomForest',
                                   line=dict(color='#f1c40f', dash='dot')))
        fig_l.add_trace(
            go.Scatter(x=subset.index, y=subset['SARIMA'], name='SARIMA', line=dict(color='#3498db', dash='dot')))
        fig_l.add_trace(
            go.Scatter(x=subset.index, y=subset['Naive'], name='Naive', line=dict(color='purple', dash='dot')))
        if 'LSTM' in subset:
            fig_l.add_trace(go.Scatter(x=subset.index, y=subset['LSTM'], name='LSTM-Δ (AI)',
                                       line=dict(color='#27ae60', width=2)))
        fig_l.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white")
        st.plotly_chart(fig_l, width="stretch")

    st.markdown("---")

    st.subheader("Podsumowanie")

    best_model = metrics_df.iloc[0]
    st.success(f"Najlepszy model w tym okresie to **{best_model['Model']}** z błędem MAE **{best_model['MAE']:.2f}**.")

    st.info("""
     **Dlaczego rekomendujemy LSTM-Δ (projekt AI, kamienie milowe 2–3)?**

     * W ujednoliconym protokole (prognozy kroczące, 5 ziaren, test Diebolda-Mariano) **LSTM-Δ jest istotnie lepszy od XGBoost**
       w okresie zmiany reżimu (MAE 0,700 vs 0,997 dla 2023–2025) i nie ma limitu ekstrapolacji (prognozuje zmiany, nie poziom).
     * XGBoost i Random Forest uczone na poziomie CPI nie potrafią przewidzieć wartości wyższych niż w zbiorze uczącym.
     * **Uczciwe zastrzeżenie:** prosta regresja liniowa na zmianach (LR-Δ) osiąga porównywalny błąd, a na nowych danych
       11.2025–10.2026 żaden model nie jest istotnie lepszy od prognozy naiwnej. Szczegóły i wyjaśnienia (XAI) – zakładka „LSTM & XAI”.
    """)

    st.markdown("---")

    # === DIAGNOSTYKA (POPRAWIONA OBSŁUGA KRÓTKICH DANYCH) ===
    with st.expander("Szczegółowa Diagnostyka SARIMA", expanded=True):
        st.markdown("### Analiza Reszt (Błędów) modelu SARIMA")

        residuals = subset['Actual'] - subset['SARIMA']
        mean_res = residuals.mean()
        std_res = residuals.std()

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Średnia błędu (Mean)", f"{mean_res:.4f}")
        d2.metric("Odchylenie (Std Dev)", f"{std_res:.4f}")

        # Test Ljung-Box
        try:
            # Automatyczne dopasowanie lagów do długości danych
            lags_lb = min(10, len(residuals) // 2 - 1)
            if lags_lb < 1: lags_lb = 1

            lb_test = acorr_ljungbox(residuals, lags=[lags_lb], return_df=True)
            lb_pvalue = lb_test['lb_pvalue'].values[0]
            lb_status = "OK (Brak korelacji)" if lb_pvalue > 0.05 else " Wykryto korelację"
            d3.metric(f"Test Ljung-Box (lag={lags_lb})", f"{lb_pvalue:.4f}", lb_status, delta_color="off")
        except:
            d3.metric("Test Ljung-Box", "Za mało danych", "Błąd")

        # Test Jarque-Bera
        try:
            jb_stat, jb_pvalue = jarque_bera(residuals)
            jb_status = "OK (Normalny)" if jb_pvalue > 0.05 else "Brak normalności"
            d4.metric("Test Jarque-Bera (p-val)", f"{jb_pvalue:.4f}", jb_status, delta_color="off")
        except:
            d4.metric("Test Jarque-Bera", "Błąd", "Error")

        # Wykresy ACF i PACF - POPRAWKA NLAGS
        st.markdown("#### Autokorelacja reszt (ACF i PACF)")
        col_acf, col_pacf = st.columns(2)

        # POPRAWKA: Dynamiczne nlags (max 20, ale nie więcej niż 50% próbki)
        n_obs = len(residuals)
        safe_lags = min(20, int(n_obs / 2) - 1)
        if safe_lags < 1: safe_lags = 1  # Zabezpieczenie

        acf_vals = acf(residuals, nlags=safe_lags)
        pacf_vals = pacf(residuals, nlags=safe_lags)
        conf_interval = 1.96 / np.sqrt(len(residuals))
        lags = np.arange(len(acf_vals))

        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=lags, y=acf_vals, name='ACF', marker_color='#3498db'))
        fig_acf.add_shape(type="rect", x0=0, x1=safe_lags, y0=-conf_interval, y1=conf_interval, fillcolor="blue",
                          opacity=0.1, line_width=0)
        fig_acf.add_hline(y=conf_interval, line_dash="dot", line_color="gray")
        fig_acf.add_hline(y=-conf_interval, line_dash="dot", line_color="gray")
        fig_acf.update_layout(title=f"ACF (lags={safe_lags})", height=300, template="plotly_white")
        col_acf.plotly_chart(fig_acf, width="stretch")

        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=lags, y=pacf_vals, name='PACF', marker_color='#9b59b6'))
        fig_pacf.add_shape(type="rect", x0=0, x1=safe_lags, y0=-conf_interval, y1=conf_interval, fillcolor="purple",
                           opacity=0.1, line_width=0)
        fig_pacf.add_hline(y=conf_interval, line_dash="dot", line_color="gray")
        fig_pacf.add_hline(y=-conf_interval, line_dash="dot", line_color="gray")
        fig_pacf.update_layout(title=f"PACF (lags={safe_lags})", height=300, template="plotly_white")
        col_pacf.plotly_chart(fig_pacf, width="stretch")

        # Histogram
        st.markdown("#### Rozkład błędów")
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(x=residuals, nbinsx=15, name='Reszty', histnorm='probability density', marker_color='#34495e',
                         opacity=0.7))
        x_range = np.linspace(min(residuals), max(residuals), 100)
        pdf = norm.pdf(x_range, mean_res, std_res)
        fig_hist.add_trace(
            go.Scatter(x=x_range, y=pdf, mode='lines', name='Rozkład Normalny', line=dict(color='red', width=2)))
        fig_hist.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("---")

    st.markdown("### Eksport wyników ewaluacji")
    csv_val = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button("Pobierz tabelę wyników (CSV)", csv_val, "wyniki_walidacji.csv", "text/csv")

# ==================================================
# TAB 3: LSTM + EKSPERYMENTY + XAI (kamienie milowe 2-3)
# ==================================================
with tab3:
    from app_lstm_tab import render as render_lstm_tab
    render_lstm_tab(history)
