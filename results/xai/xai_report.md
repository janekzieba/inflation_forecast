# Raport XAI i analiza błędów (wygenerowane przez xai_lstm.py)

## 1. Które miesiące historii decydują o prognozie?

- Podział A, LSTM-Δ: najważniejsze opóźnienia wg IG: lag 1 (0.130), lag 2 (0.013), lag 3 (0.011), lag 4 (0.009), lag 5 (0.007). XGBoost (SHAP): lag_1 (1.372), lag_12 (0.118), lag_2 (0.100).
- Podział B, LSTM-Δ: najważniejsze opóźnienia wg IG: lag 1 (0.158), lag 12 (0.062), lag 8 (0.061), lag 7 (0.055), lag 9 (0.050). XGBoost (SHAP): lag_1 (4.089), lag_3 (0.282), lag_12 (0.133).

## 2. Gdzie i dlaczego model się myli?

### Obciążenie w podokresach (h=1, średni błąd ze znakiem; > 0 = model zaniża)

|                     |   całość |   dezinflacja 03.2023- |   ekstrapolacja 2021-2023 |   stabilny 2015-2019 |   szok 2020-02.2023 |
|:--------------------|---------:|-----------------------:|--------------------------:|---------------------:|--------------------:|
| ('A', 'LSTM-L')     |    1.139 |                  1.041 |                     4.127 |                0.001 |               3.019 |
| ('A', 'LSTM-MO')    |    0.066 |                 -0.309 |                     0.157 |                0.052 |               0.406 |
| ('A', 'LSTM-Δ')     |    0.03  |                 -0.393 |                     0.028 |                0.094 |               0.283 |
| ('A', 'LSTM-Δ+kal') |    0.026 |                 -0.379 |                     0.046 |                0.082 |               0.281 |
| ('A', 'Naive')      |    0.029 |                 -0.488 |                     0.106 |                0.073 |               0.395 |
| ('A', 'SARIMA')     |    0.018 |                 -0.564 |                     0.238 |                0.083 |               0.405 |
| ('A', 'XGBoost')    |    1.847 |                  2.359 |                     6.521 |               -0.009 |               4.347 |
| ('A', 'XGBoost-Δ')  |    0.135 |                 -0.197 |                     0.322 |                0.088 |               0.487 |
| ('B', 'LSTM-L')     |    0.836 |                  0.469 |                     2.414 |              nan     |             nan     |
| ('B', 'LSTM-MO')    |   -0.328 |                 -0.407 |                    -0.723 |              nan     |             nan     |
| ('B', 'LSTM-Δ')     |   -0.284 |                 -0.363 |                    -0.693 |              nan     |             nan     |
| ('B', 'LSTM-Δ+kal') |   -0.306 |                 -0.393 |                    -0.757 |              nan     |             nan     |
| ('B', 'Naive')      |   -0.406 |                 -0.488 |                    -0.867 |              nan     |             nan     |
| ('B', 'SARIMA')     |   -0.36  |                 -0.445 |                    -0.61  |              nan     |             nan     |
| ('B', 'XGBoost')    |   -0.395 |                 -0.44  |                    -1.435 |              nan     |             nan     |
| ('B', 'XGBoost-Δ')  |   -0.519 |                 -0.558 |                    -1.261 |              nan     |             nan     |

### Punkty zwrotne (h=1)

| split   | model      |   h |   DA_turning |   DA_other |   MAE_turning |   MAE_other |   n_turning |
|:--------|:-----------|----:|-------------:|-----------:|--------------:|------------:|------------:|
| A       | LSTM-L     |   1 |        55.56 |      40.78 |          1.36 |        1.41 |          27 |
| A       | LSTM-MO    |   1 |        51.85 |      62.14 |          0.48 |        0.46 |          27 |
| A       | LSTM-Δ     |   1 |        51.85 |      64.08 |          0.49 |        0.44 |          27 |
| A       | LSTM-Δ+kal |   1 |        40.74 |      64.08 |          0.5  |        0.44 |          27 |
| A       | Naive      |   1 |         7.41 |       7.77 |          0.47 |        0.49 |          27 |
| A       | SARIMA     |   1 |        62.96 |      52.43 |          0.51 |        0.49 |          27 |
| A       | XGBoost    |   1 |        44.44 |      39.81 |          1.94 |        2.24 |          27 |
| A       | XGBoost-Δ  |   1 |        55.56 |      56.31 |          0.42 |        0.45 |          27 |
| B       | LSTM-L     |   1 |        33.33 |      60.71 |          2.29 |        0.99 |           6 |
| B       | LSTM-MO    |   1 |        16.67 |      67.86 |          0.89 |        0.62 |           6 |
| B       | LSTM-Δ     |   1 |        16.67 |      60.71 |          1.06 |        0.62 |           6 |
| B       | LSTM-Δ+kal |   1 |         0    |      64.29 |          1.09 |        0.65 |           6 |
| B       | Naive      |   1 |        33.33 |      14.29 |          0.8  |        0.69 |           6 |
| B       | SARIMA     |   1 |        16.67 |      67.86 |          1.15 |        0.62 |           6 |
| B       | XGBoost    |   1 |        50    |      32.14 |          0.7  |        1.06 |           6 |
| B       | XGBoost-Δ  |   1 |        50    |      46.43 |          1.01 |        0.86 |           6 |

### Ekstrapolacja poza maksimum treningu (Podział A, max = 107.4)

| model      |   train_max |   n_above |   share_pred_above_max |   max_pred |   MAE_above |   MAE_within |
|:-----------|------------:|----------:|-----------------------:|-----------:|------------:|-------------:|
| Naive      |       107.4 |        23 |                 95.652 |    118.4   |       1.139 |        0.35  |
| SARIMA     |       107.4 |        23 |                 95.652 |    119.102 |       1.031 |        0.379 |
| XGBoost    |       107.4 |        23 |                  0     |    104.627 |       9.486 |        0.608 |
| XGBoost-Δ  |       107.4 |        23 |                 95.652 |    118.378 |       1.028 |        0.32  |
| LSTM-L     |       107.4 |        23 |                 52.174 |    107.608 |       6.188 |        0.373 |
| LSTM-Δ     |       107.4 |        23 |                 95.652 |    118.942 |       1.016 |        0.328 |
| LSTM-Δ+kal |       107.4 |        23 |                 95.652 |    118.818 |       1.022 |        0.331 |
| LSTM-MO    |       107.4 |        23 |                 95.652 |    118.434 |       1.078 |        0.332 |

### 10 największych błędów najlepszego LSTM (h ∈ {1,3})

| split   | model   |   h | origin              | target_date         |   y_true |   y_pred |    err |
|:--------|:--------|----:|:--------------------|:--------------------|---------:|---------:|-------:|
| A       | LSTM-Δ  |   3 | 2023-02-01 00:00:00 | 2023-05-01 00:00:00 |    113   |  119.258 | -6.258 |
| A       | LSTM-Δ  |   3 | 2022-02-01 00:00:00 | 2022-05-01 00:00:00 |    113.9 |  108.325 |  5.575 |
| A       | LSTM-Δ  |   3 | 2023-03-01 00:00:00 | 2023-06-01 00:00:00 |    111.5 |  116.039 | -4.539 |
| A       | LSTM-Δ  |   3 | 2023-12-01 00:00:00 | 2024-03-01 00:00:00 |    102   |  105.802 | -3.802 |
| A       | LSTM-Δ  |   3 | 2023-07-01 00:00:00 | 2023-10-01 00:00:00 |    106.6 |  110.299 | -3.699 |
| A       | LSTM-Δ  |   3 | 2023-11-01 00:00:00 | 2024-02-01 00:00:00 |    102.8 |  106.395 | -3.595 |
| A       | LSTM-Δ  |   3 | 2023-04-01 00:00:00 | 2023-07-01 00:00:00 |    110.8 |  114.331 | -3.531 |
| A       | LSTM-Δ  |   3 | 2022-03-01 00:00:00 | 2022-06-01 00:00:00 |    115.5 |  112.034 |  3.466 |
| A       | LSTM-Δ  |   3 | 2023-08-01 00:00:00 | 2023-11-01 00:00:00 |    106.6 |  109.594 | -2.994 |
| A       | LSTM-Δ  |   1 | 2023-02-01 00:00:00 | 2023-03-01 00:00:00 |    116.1 |  118.942 | -2.842 |
| B       | LSTM-Δ  |   3 | 2023-02-01 00:00:00 | 2023-05-01 00:00:00 |    113   |  120.383 | -7.383 |
| B       | LSTM-Δ  |   3 | 2023-03-01 00:00:00 | 2023-06-01 00:00:00 |    111.5 |  115.764 | -4.264 |
| B       | LSTM-Δ  |   3 | 2023-07-01 00:00:00 | 2023-10-01 00:00:00 |    106.6 |  110.075 | -3.475 |
| B       | LSTM-Δ  |   3 | 2023-12-01 00:00:00 | 2024-03-01 00:00:00 |    102   |  105.332 | -3.332 |
| B       | LSTM-Δ  |   1 | 2023-02-01 00:00:00 | 2023-03-01 00:00:00 |    116.1 |  119.327 | -3.227 |
| B       | LSTM-Δ  |   3 | 2023-04-01 00:00:00 | 2023-07-01 00:00:00 |    110.8 |  114.017 | -3.217 |
| B       | LSTM-Δ  |   3 | 2023-11-01 00:00:00 | 2024-02-01 00:00:00 |    102.8 |  105.818 | -3.018 |
| B       | LSTM-Δ  |   3 | 2024-06-01 00:00:00 | 2024-09-01 00:00:00 |    104.9 |  102.043 |  2.857 |
| B       | LSTM-Δ  |   3 | 2023-08-01 00:00:00 | 2023-11-01 00:00:00 |    106.6 |  109.369 | -2.769 |
| B       | LSTM-Δ  |   3 | 2023-01-01 00:00:00 | 2023-04-01 00:00:00 |    114.7 |  117.251 | -2.551 |

## 3. Niepewność (MC-dropout vs SARIMA)

| split   | model      |   h | method                  |   coverage95 |   mean_width |   corr_width_abs_err |
|:--------|:-----------|----:|:------------------------|-------------:|-------------:|---------------------:|
| A       | LSTM-L     |   1 | MC-dropout              |       53.846 |        1.199 |                0.666 |
| A       | LSTM-L     |   1 | konformalny (walidacja) |       63.846 |        1.038 |              nan     |
| A       | LSTM-L     |   3 | MC-dropout              |       39.062 |        1.559 |                0.147 |
| A       | LSTM-L     |   3 | konformalny (walidacja) |       59.375 |        2.11  |              nan     |
| A       | LSTM-L     |   6 | MC-dropout              |       34.4   |        1.849 |               -0.192 |
| A       | LSTM-L     |   6 | konformalny (walidacja) |       47.2   |        2.249 |              nan     |
| A       | LSTM-L     |  12 | MC-dropout              |       26.05  |        2.379 |               -0.213 |
| A       | LSTM-L     |  12 | konformalny (walidacja) |       43.697 |        3.38  |              nan     |
| A       | LSTM-MO    |   1 | MC-dropout              |       17.692 |        0.195 |                0.393 |
| A       | LSTM-MO    |   1 | konformalny (walidacja) |       72.308 |        1.077 |              nan     |
| A       | LSTM-MO    |   3 | MC-dropout              |       18.75  |        0.422 |                0.487 |
| A       | LSTM-MO    |   3 | konformalny (walidacja) |       67.188 |        2.498 |              nan     |
| A       | LSTM-MO    |   6 | MC-dropout              |       19.2   |        0.781 |                0.515 |
| A       | LSTM-MO    |   6 | konformalny (walidacja) |       58.4   |        3.339 |              nan     |
| A       | LSTM-MO    |  12 | MC-dropout              |       24.37  |        1.334 |                0.353 |
| A       | LSTM-MO    |  12 | konformalny (walidacja) |       53.782 |        3.968 |              nan     |
| A       | LSTM-Δ     |   1 | MC-dropout              |       11.538 |        0.136 |                0.38  |
| A       | LSTM-Δ     |   1 | konformalny (walidacja) |       72.308 |        0.988 |              nan     |
| A       | LSTM-Δ     |   3 | MC-dropout              |       11.719 |        0.263 |                0.484 |
| A       | LSTM-Δ     |   3 | konformalny (walidacja) |       65.625 |        2.219 |              nan     |
| A       | LSTM-Δ     |   6 | MC-dropout              |        8     |        0.376 |                0.522 |
| A       | LSTM-Δ     |   6 | konformalny (walidacja) |       47.2   |        2.68  |              nan     |
| A       | LSTM-Δ     |  12 | MC-dropout              |        5.042 |        0.555 |                0.288 |
| A       | LSTM-Δ     |  12 | konformalny (walidacja) |       32.773 |        2.958 |              nan     |
| A       | LSTM-Δ+kal |   1 | MC-dropout              |       10.769 |        0.137 |                0.342 |
| A       | LSTM-Δ+kal |   1 | konformalny (walidacja) |       71.538 |        0.986 |              nan     |
| A       | LSTM-Δ+kal |   3 | MC-dropout              |       10.938 |        0.267 |                0.386 |
| A       | LSTM-Δ+kal |   3 | konformalny (walidacja) |       60.938 |        1.969 |              nan     |
| A       | LSTM-Δ+kal |   6 | MC-dropout              |       10.4   |        0.402 |                0.405 |
| A       | LSTM-Δ+kal |   6 | konformalny (walidacja) |       46.4   |        2.375 |              nan     |
| A       | LSTM-Δ+kal |  12 | MC-dropout              |        4.202 |        0.625 |                0.041 |
| A       | LSTM-Δ+kal |  12 | konformalny (walidacja) |       37.815 |        2.972 |              nan     |
| A       | SARIMA     |   1 | analityczny             |       82.308 |        1.657 |               -0.146 |
| A       | SARIMA     |   3 | analityczny             |       71.094 |        3.587 |                0.041 |
| A       | SARIMA     |   6 | analityczny             |       67.2   |        5.224 |                0.033 |
| A       | SARIMA     |  12 | analityczny             |       63.025 |        6.673 |               -0.149 |
| B       | LSTM-L     |   1 | MC-dropout              |       32.353 |        0.808 |                0.765 |
| B       | LSTM-L     |   1 | konformalny (walidacja) |       97.059 |       13.307 |              nan     |
| B       | LSTM-L     |   3 | MC-dropout              |       21.875 |        1.036 |                0.567 |
| B       | LSTM-L     |   3 | konformalny (walidacja) |      100     |       19.591 |              nan     |
| B       | LSTM-L     |   6 | MC-dropout              |       20.69  |        1.05  |                0.123 |
| B       | LSTM-L     |   6 | konformalny (walidacja) |      100     |       23.445 |              nan     |
| B       | LSTM-L     |  12 | MC-dropout              |       26.087 |        1.201 |                0.438 |
| B       | LSTM-L     |  12 | konformalny (walidacja) |      100     |       27.899 |              nan     |
| B       | LSTM-MO    |   1 | MC-dropout              |        2.941 |        0.072 |                0.031 |
| B       | LSTM-MO    |   1 | konformalny (walidacja) |       73.529 |        2.388 |              nan     |
| B       | LSTM-MO    |   3 | MC-dropout              |        3.125 |        0.129 |               -0.053 |
| B       | LSTM-MO    |   3 | konformalny (walidacja) |       87.5   |        7.276 |              nan     |
| B       | LSTM-MO    |   6 | MC-dropout              |        0     |        0.18  |               -0.01  |
| B       | LSTM-MO    |   6 | konformalny (walidacja) |       82.759 |       12.306 |              nan     |
| B       | LSTM-MO    |  12 | MC-dropout              |        0     |        0.23  |               -0.142 |
| B       | LSTM-MO    |  12 | konformalny (walidacja) |       82.609 |       22.849 |              nan     |
| B       | LSTM-Δ     |   1 | MC-dropout              |        0     |        0.092 |                0.608 |
| B       | LSTM-Δ     |   1 | konformalny (walidacja) |       73.529 |        2.184 |              nan     |
| B       | LSTM-Δ     |   3 | MC-dropout              |        3.125 |        0.203 |                0.654 |
| B       | LSTM-Δ     |   3 | konformalny (walidacja) |       84.375 |        6.116 |              nan     |
| B       | LSTM-Δ     |   6 | MC-dropout              |        0     |        0.311 |                0.46  |
| B       | LSTM-Δ     |   6 | konformalny (walidacja) |       86.207 |       12.581 |              nan     |
| B       | LSTM-Δ     |  12 | MC-dropout              |        0     |        0.459 |                0.521 |
| B       | LSTM-Δ     |  12 | konformalny (walidacja) |       86.957 |       21.398 |              nan     |
| B       | LSTM-Δ+kal |   1 | MC-dropout              |        0     |        0.1   |                0.44  |
| B       | LSTM-Δ+kal |   1 | konformalny (walidacja) |       73.529 |        2.41  |              nan     |
| B       | LSTM-Δ+kal |   3 | MC-dropout              |        9.375 |        0.218 |                0.32  |
| B       | LSTM-Δ+kal |   3 | konformalny (walidacja) |       84.375 |        6.336 |              nan     |
| B       | LSTM-Δ+kal |   6 | MC-dropout              |        0     |        0.347 |               -0.005 |
| B       | LSTM-Δ+kal |   6 | konformalny (walidacja) |       86.207 |       12.883 |              nan     |
| B       | LSTM-Δ+kal |  12 | MC-dropout              |        4.348 |        0.59  |               -0.177 |
| B       | LSTM-Δ+kal |  12 | konformalny (walidacja) |       82.609 |       21.9   |              nan     |
| B       | SARIMA     |   1 | analityczny             |       70.588 |        1.886 |               -0.194 |
| B       | SARIMA     |   3 | analityczny             |       71.875 |        4.331 |                0.124 |
| B       | SARIMA     |   6 | analityczny             |       55.172 |        6.917 |               -0.01  |
| B       | SARIMA     |  12 | analityczny             |       47.826 |       10.337 |               -0.012 |