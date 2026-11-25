# --- ARCHIWUM (praca inżynierska) – uruchamiaj z głównego folderu: python archiwum/skrypty/<plik>.py
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_sys.path.insert(0, _ROOT)                                   # data_loader z głównego folderu
_os.makedirs(_os.path.join(_ROOT, "archiwum", "wyniki"), exist_ok=True)
_os.chdir(_os.path.join(_ROOT, "archiwum", "wyniki"))        # wykresy/CSV zapisują się w archiwum/wyniki
# ---
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from data_loader import load_data


# ===== 1) CECHY (tak jak u Ciebie) =====
def create_features(df):
    df = df.copy()
    df['lag_1'] = df['CPI'].shift(1)
    df['lag_2'] = df['CPI'].shift(2)
    df['lag_3'] = df['CPI'].shift(3)
    df['lag_6'] = df['CPI'].shift(6)
    df['lag_12'] = df['CPI'].shift(12)
    df['roll_mean_3'] = df['CPI'].shift(1).rolling(window=3).mean()
    df['roll_mean_6'] = df['CPI'].shift(1).rolling(window=6).mean()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    return df


def mae_rmse(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ===== 2) USTAWIENIA BACKTESTU =====
HORIZON_MONTHS = 24          # długość okna testowego dla każdego splitu
SHOCK_Q = 0.90               # próg "szoku": 90 percentyl |ΔCPI| z treningu
SPLITS = [
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
    "2022-01-01",
    "2023-01-01",
]

# Parametry modeli (startowo)
XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective="reg:squarederror",
)
RF_PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)


# ===== 3) DANE =====
df_raw = load_data()
df_raw = df_raw.sort_index()

df_feat = create_features(df_raw).dropna()

features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'roll_mean_3', 'roll_mean_6', 'month', 'quarter']
target = 'CPI'

X_all = df_feat[features]
y_all = df_feat[target]

# do definicji "szoku"
abs_delta_all = df_raw['CPI'].diff().abs()


# ===== 4) ROLLING-ORIGIN + SZOKI =====
rows = []

for split_str in SPLITS:
    split = pd.Timestamp(split_str)
    test_end = split + pd.DateOffset(months=HORIZON_MONTHS)

    # train: < split, test: [split, test_end)
    train_mask = X_all.index < split
    test_mask = (X_all.index >= split) & (X_all.index < test_end)

    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

    if len(X_test) == 0 or len(X_train) == 0:
        continue

    # próg szoku liczony TYLKO z treningu
    train_abs_delta = abs_delta_all.loc[abs_delta_all.index < split].dropna()
    shock_thr = train_abs_delta.quantile(SHOCK_Q)

    test_abs_delta = abs_delta_all.reindex(y_test.index)
    shock_mask = (test_abs_delta > shock_thr).fillna(False)
    normal_mask = ~shock_mask

    # modele
    xgb = XGBRegressor(**XGB_PARAMS)
    rf = RandomForestRegressor(**RF_PARAMS)

    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    pred_xgb = pd.Series(xgb.predict(X_test), index=y_test.index)
    pred_rf  = pd.Series(rf.predict(X_test), index=y_test.index)

    # overall
    mae_xgb, rmse_xgb = mae_rmse(y_test, pred_xgb)
    mae_rf, rmse_rf   = mae_rmse(y_test, pred_rf)

    # shock / normal (jeśli są)
    def safe_metrics(mask, y_true, y_pred):
        if mask.sum() == 0:
            return np.nan, np.nan, 0
        m_mae, m_rmse = mae_rmse(y_true[mask], y_pred[mask])
        return m_mae, m_rmse, int(mask.sum())

    mae_xgb_s, rmse_xgb_s, n_s = safe_metrics(shock_mask, y_test, pred_xgb)
    mae_rf_s,  rmse_rf_s,  _   = safe_metrics(shock_mask, y_test, pred_rf)

    mae_xgb_n, rmse_xgb_n, n_n = safe_metrics(normal_mask, y_test, pred_xgb)
    mae_rf_n,  rmse_rf_n,  _   = safe_metrics(normal_mask, y_test, pred_rf)

    rows.append({
        "split_start": split.date(),
        "test_len": len(y_test),
        "shock_thr_abs_dCPI": float(shock_thr),
        "n_shock": n_s,
        "n_normal": n_n,

        "MAE_XGB": mae_xgb, "RMSE_XGB": rmse_xgb,
        "MAE_RF": mae_rf,   "RMSE_RF": rmse_rf,

        "MAE_XGB_shock": mae_xgb_s, "RMSE_XGB_shock": rmse_xgb_s,
        "MAE_RF_shock": mae_rf_s,   "RMSE_RF_shock": rmse_rf_s,

        "MAE_XGB_normal": mae_xgb_n, "RMSE_XGB_normal": rmse_xgb_n,
        "MAE_RF_normal": mae_rf_n,   "RMSE_RF_normal": rmse_rf_n,
    })

results = pd.DataFrame(rows)
print("\n=== ROLLING-ORIGIN RESULTS ===")
print(results.round(4).to_string(index=False))

print("\n=== PODSUMOWANIE (mean ± std) ===")
summary_cols = ["MAE_XGB", "MAE_RF", "RMSE_XGB", "RMSE_RF",
                "MAE_XGB_shock", "MAE_RF_shock", "MAE_XGB_normal", "MAE_RF_normal"]
summary = results[summary_cols].agg(["mean", "std"]).T
print(summary.round(4).to_string())

results.to_csv("rolling_shock_eval.csv", index=False)
print("\n>>> Zapisano: rolling_shock_eval.csv")
