"""
Wspólny protokół ewaluacji (kamień milowy 2/3).

- chronologiczne podziały A / B,
- prognozy z kroczącym punktem startowym (rolling origin) dla h = 1..12,
- metryki: MAE, RMSE, MAPE, MASE, RelMAE (względem Naive), trafność kierunku, pokrycie przedziałów,
- test Diebolda-Mariano (z poprawką Harveya-Leybourne'a-Newbolda),
- modele bazowe: Naive, Seasonal Naive, LR, SARIMA, XGBoost, RF, k-NN, SVR (poziom i przyrost).

Format prognoz (long): model, seed, origin, h, target_date, y_true, y_pred, [lo, hi]
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from data_loader import load_data

# ------------------------------------------------------------------
# Konfiguracja eksperymentu
# ------------------------------------------------------------------
SPLITS = {
    "A": pd.Timestamp("2015-01-01"),   # długi horyzont: trening 2001-2014
    "B": pd.Timestamp("2023-01-01"),   # zmiana reżimu: trening 2001-2022
}
H_MAX = 12
HORIZONS = [1, 3, 6, 12]
PERIODS = {
    "całość": (None, None),
    "stabilny 2015-2019": ("2015-01-01", "2019-12-01"),
    "szok 2020-02.2023": ("2020-01-01", "2023-02-01"),
    "dezinflacja 03.2023-": ("2023-03-01", None),
    "ekstrapolacja 2021-2023": ("2021-01-01", "2023-12-01"),
}
VAL_MONTHS = 24


def load_series() -> pd.Series:
    df = load_data()
    s = df["CPI"].astype(float)
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp()
    s = s.asfreq("MS")
    if s.isna().any():
        raise ValueError("Szereg CPI ma luki – uruchom ponownie data_source.py")
    return s


def origins_for_split(y: pd.Series, split: str) -> np.ndarray:
    """Pozycje punktów startowych: od ostatniej obserwacji treningowej do przedostatniej obserwacji."""
    start = y.index.get_loc(SPLITS[split])
    return np.arange(start - 1, len(y) - 1)


def to_long(model, seed, y, origins, preds, lo=None, hi=None) -> pd.DataFrame:
    """preds: macierz (n_origins, H) z prognozami poziomu CPI."""
    rows = []
    n = len(y)
    vals = y.values
    for r, i in enumerate(origins):
        for k in range(1, preds.shape[1] + 1):
            j = i + k
            if j >= n:
                break
            row = dict(model=model, seed=seed, origin=y.index[i], h=k, target_date=y.index[j],
                       y_origin=vals[i], y_true=vals[j], y_pred=float(preds[r, k - 1]))
            if lo is not None:
                row["lo"] = float(lo[r, k - 1])
                row["hi"] = float(hi[r, k - 1])
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Metryki
# ------------------------------------------------------------------
def mase_scale(y_train: pd.Series, h: int) -> float:
    """Skala MASE: średni |y_t - y_{t-h}| na zbiorze treningowym (naiwna prognoza h-krokowa in-sample)."""
    v = y_train.values
    return float(np.mean(np.abs(v[h:] - v[:-h])))


def compute_metrics(df: pd.DataFrame, y_train: pd.Series) -> dict:
    e = df["y_true"].values - df["y_pred"].values
    h = int(df["h"].iloc[0])
    out = dict(
        n=len(df),
        MAE=np.mean(np.abs(e)),
        RMSE=np.sqrt(np.mean(e ** 2)),
        MAPE=np.mean(np.abs(e) / np.abs(df["y_true"].values)) * 100,
        MASE=np.mean(np.abs(e)) / mase_scale(y_train, h),
    )
    true_dir = np.sign(df["y_true"].values - df["y_origin"].values)
    pred_dir = np.sign(df["y_pred"].values - df["y_origin"].values)
    out["DA"] = np.mean(true_dir == pred_dir) * 100
    if "lo" in df and df["lo"].notna().all():
        out["coverage95"] = np.mean((df["y_true"] >= df["lo"]) & (df["y_true"] <= df["hi"])) * 100
        out["PI_width"] = np.mean(df["hi"] - df["lo"])
    else:
        out["coverage95"] = np.nan
        out["PI_width"] = np.nan
    if "lo_conf" in df and df["lo_conf"].notna().all():
        out["coverage95_conf"] = np.mean((df["y_true"] >= df["lo_conf"]) & (df["y_true"] <= df["hi_conf"])) * 100
        out["PI_width_conf"] = np.mean(df["hi_conf"] - df["lo_conf"])
    else:
        out["coverage95_conf"] = np.nan
        out["PI_width_conf"] = np.nan
    return out


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    a, b = PERIODS[period]
    m = pd.Series(True, index=df.index)
    if a:
        m &= df["target_date"] >= pd.Timestamp(a)
    if b:
        m &= df["target_date"] <= pd.Timestamp(b)
    return df[m]


def metrics_table(fc: pd.DataFrame, y: pd.Series, split: str) -> pd.DataFrame:
    y_train = y.loc[y.index < SPLITS[split]]
    rows = []
    for (model, seed), g in fc.groupby(["model", "seed"]):
        for h in HORIZONS:
            gh = g[g["h"] == h]
            for p in PERIODS:
                gp = filter_period(gh, p)
                if len(gp) < 3:
                    continue
                m = compute_metrics(gp, y_train)
                rows.append(dict(split=split, model=model, seed=seed, h=h, period=p, **m))
    mt = pd.DataFrame(rows)
    # RelMAE: MAE modelu / MAE modelu Naive na tych samych punktach (< 1 = lepiej niż Naive)
    naive = mt[mt["model"] == "Naive"].set_index(["h", "period"])["MAE"]
    mt["RelMAE"] = mt.apply(lambda r: r["MAE"] / naive.get((r["h"], r["period"]), np.nan), axis=1)
    return mt


def summarize(mt: pd.DataFrame) -> pd.DataFrame:
    """Średnia i odchylenie std po ziarnach losowych."""
    cols = ["MAE", "RMSE", "MAPE", "MASE", "RelMAE", "DA", "coverage95", "PI_width", "coverage95_conf", "PI_width_conf"]
    g = mt.groupby(["split", "model", "h", "period"])
    mean = g[cols].mean()
    std = g[["MAE", "RMSE"]].std().fillna(0.0).add_suffix("_std")
    n_seeds = g["seed"].nunique().rename("n_seeds")
    n = g["n"].first()
    return pd.concat([mean, std, n_seeds, n], axis=1).reset_index()


# ------------------------------------------------------------------
# Test Diebolda-Mariano
# ------------------------------------------------------------------
def dm_test(e1: np.ndarray, e2: np.ndarray, h: int, loss: str = "abs"):
    """H0: równa trafność. Statystyka < 0 => model 1 lepszy.
    Zwraca (DM_HLN, p dwustronne, p jednostronne [model 1 lepszy])."""
    e1, e2 = np.asarray(e1), np.asarray(e2)
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = e1 ** 2 - e2 ** 2
    T = len(d)
    if T < 5 or np.allclose(d, 0):
        return np.nan, np.nan, np.nan
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:T - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        var = gamma[0] / T
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p2 = 2 * stats.t.sf(np.abs(stat), df=T - 1)
    p1 = stats.t.cdf(stat, df=T - 1)
    return stat, p2, p1


def dm_table(fc: pd.DataFrame, split: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Test DM dla par (model, odniesienie). Dla modeli z wieloma ziarnami używana jest prognoza uśredniona
    po ziarnach (ensemble), co odpowiada stabilnej wersji modelu."""
    avg = fc.groupby(["model", "h", "target_date", "origin"], as_index=False)[["y_true", "y_pred"]].mean()
    rows = []
    for m1, m2 in pairs:
        for h in HORIZONS:
            for p in PERIODS:
                a = filter_period(avg[(avg.model == m1) & (avg.h == h)], p).set_index("target_date")
                b = filter_period(avg[(avg.model == m2) & (avg.h == h)], p).set_index("target_date")
                idx = a.index.intersection(b.index)
                if len(idx) < 6:
                    continue
                e1 = (a.loc[idx, "y_true"] - a.loc[idx, "y_pred"]).values
                e2 = (b.loc[idx, "y_true"] - b.loc[idx, "y_pred"]).values
                stat, p2, p1 = dm_test(e1, e2, h)
                rows.append(dict(split=split, model=m1, reference=m2, h=h, period=p, n=len(idx),
                                 MAE_model=np.mean(np.abs(e1)), MAE_ref=np.mean(np.abs(e2)),
                                 DM=stat, p_two_sided=p2, p_model_better=p1))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Modele bazowe
# ------------------------------------------------------------------
LAGS = [1, 2, 3, 6, 12]
FEATURES = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "roll_mean_3", "roll_mean_6", "month", "quarter"]


def features_from_buffer(buf: np.ndarray, months: np.ndarray) -> np.ndarray:
    """buf: (n, >=12) ostatnie wartości szeregu (ostatnia kolumna = t). months: miesiąc celu t+1.
    Cechy identyczne jak create_features z pracy inżynierskiej."""
    cols = [buf[:, -l] for l in LAGS]
    cols.append(buf[:, -3:].mean(axis=1))
    cols.append(buf[:, -6:].mean(axis=1))
    cols.append(months.astype(float))
    cols.append(((months - 1) // 3 + 1).astype(float))
    return np.column_stack(cols)


def build_train_matrix(s: pd.Series, end_pos: int):
    """Macierz cech dla celów s[t], t < end_pos (bez wycieku: cechy z t-1 i wcześniej)."""
    v = s.values
    X, Y = [], []
    for t in range(12, end_pos):
        if np.isnan(v[t - 12:t + 1]).any():
            continue
        X.append(features_from_buffer(v[t - 12:t][None, :], np.array([s.index[t].month]))[0])
        Y.append(v[t])
    return np.array(X), np.array(Y)


def recursive_forecast(model, s: pd.Series, origins: np.ndarray, H: int) -> np.ndarray:
    """Prognoza rekurencyjna wektorowo dla wszystkich punktów startowych naraz. Zwraca (n_origins, H) w przestrzeni s."""
    v = s.values
    buf = np.stack([v[i - 11:i + 1] for i in origins])
    out = np.zeros((len(origins), H))
    for k in range(H):
        months = np.array([(s.index[i] + pd.DateOffset(months=k + 1)).month for i in origins])
        X = features_from_buffer(buf, months)
        p = model.predict(X)
        out[:, k] = p
        buf = np.column_stack([buf[:, 1:], p])
    return out


def make_ml_models(seed: int = 42):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
    return {
        "LR": LinearRegression(),
        "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=seed, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=500, max_depth=10, random_state=seed, n_jobs=-1),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "SVR": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100)),
    }


def run_ml_model(name, model, y: pd.Series, split: str, target: str = "level"):
    """target='level' – jak w pracy inżynierskiej; target='delta' – model uczy się Δy na cechach z Δy."""
    split_pos = y.index.get_loc(SPLITS[split])
    origins = origins_for_split(y, split)
    s = y if target == "level" else y.diff()
    X, Y = build_train_matrix(s, split_pos)
    t0 = time.time()
    model.fit(X, Y)
    t_train = time.time() - t0
    t0 = time.time()
    p = recursive_forecast(model, s, origins, H_MAX)
    t_inf = time.time() - t0
    if target == "delta":
        p = y.values[origins][:, None] + np.cumsum(p, axis=1)
    return p, origins, t_train, t_inf, model


def run_naive(y, split):
    origins = origins_for_split(y, split)
    p = np.repeat(y.values[origins][:, None], H_MAX, axis=1)
    return p, origins


def run_seasonal_naive(y, split):
    origins = origins_for_split(y, split)
    v = y.values
    p = np.array([[v[i + k - 12] for k in range(1, H_MAX + 1)] for i in origins])
    return p, origins


def run_sarima(y, split, order=(2, 0, 1), seasonal_order=(2, 1, 0, 12)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    split_pos = y.index.get_loc(SPLITS[split])
    origins = origins_for_split(y, split)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y.iloc[:split_pos], order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    t_train = time.time() - t0
    t0 = time.time()
    P, L, U = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in origins:
            r = res.apply(y.iloc[:i + 1])  # te same parametry, filtr Kalmana na danych do t
            f = r.get_forecast(H_MAX)
            ci = f.conf_int(alpha=0.05).values
            P.append(f.predicted_mean.values)
            L.append(ci[:, 0])
            U.append(ci[:, 1])
    t_inf = time.time() - t0
    return np.array(P), np.array(L), np.array(U), origins, t_train, t_inf
