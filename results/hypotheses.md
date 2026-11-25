# Weryfikacja hipotez (wygenerowane automatycznie przez experiments.py)

Najlepszy wariant LSTM wybrano na podstawie **walidacji kroczącej w zbiorze treningowym** (bez podglądania testu):
- Podział A: **LSTM-Δ**, hiperparametry: `{"window": 12, "hidden": 32, "dropout": 0.3, "lr": 0.005}`
- Podział B: **LSTM-Δ**, hiperparametry: `{"window": 12, "hidden": 64, "dropout": 0.1, "lr": 0.001}`

## H1: co najmniej jeden model uczony jest istotnie lepszy od Naive (DM, α = 0,05)

- Podział A: **POTWIERDZONA** – LR-Δ (h=1, p=0.001), LSTM-Δ (h=1, p=0.004), LSTM-Δ+kal (h=1, p=0.007), LSTM-MO (h=1, p=0.030)
- Podział B: **NIE POTWIERDZONA** – żaden model nie jest istotnie lepszy od Naive

## H2: w okresie zmiany reżimu (Podział B) LSTM ma niższe MAE niż XGBoost

Punkt odniesienia z pracy inżynierskiej: XGBoost MAE = 0,997 (prognoza 1-krokowa).

| h | LSTM-L | LSTM-Δ | LSTM-Δ+kal | LSTM-MO | XGBoost | XGBoost-Δ | wynik (najlepszy LSTM) |
|---|---|---|---|---|---|---|---|
| 1 | 1.220 | 0.700 | 0.728 | 0.672 | 0.997 | 0.883 | TAK |
| 3 | 1.815 | 1.768 | 1.858 | 1.734 | 2.144 | 1.831 | TAK |
| 6 | 1.985 | 3.524 | 3.642 | 3.381 | 3.842 | 3.247 | TAK |
| 12 | 1.588 | 5.365 | 5.712 | 5.139 | 5.672 | 4.830 | TAK |

## H3: prognozowanie przyrostów zmniejsza błąd modeli nieliniowych w okresie 2021-2023

Okres 'ekstrapolacja 2021-2023', Podział A (wartości CPI poza zakresem treningu 2001-2014, max 107,4).

| h | LSTM-L | LSTM-Δ | XGBoost | XGBoost-Δ | LSTM: Δ lepszy? | XGB: Δ lepszy? |
|---|---|---|---|---|---|---|
| 1 | 4.168 | 0.839 | 6.522 | 0.846 | TAK | TAK |
| 3 | 5.166 | 2.044 | 7.395 | 2.047 | TAK | TAK |
| 6 | 6.073 | 3.888 | 8.160 | 3.975 | TAK | TAK |
| 12 | 7.135 | 6.580 | 8.565 | 7.115 | TAK | TAK |

## H4: przewaga LSTM nad modelami bazowymi rośnie z horyzontem

Podział A – LSTM-Δ: RelMAE (MAE / MAE Naive; < 1 = lepiej niż Naive)

| h | RelMAE LSTM | RelMAE SARIMA | RelMAE XGBoost | MAE najlepszego modelu bazowego | MAE LSTM |
|---|---|---|---|---|---|
| 1 | 0.924 | 1.010 | 4.454 | 0.448 (LR) | 0.452 |
| 3 | 0.963 | 1.062 | 2.368 | 1.134 (LR) | 1.124 |
| 6 | 1.029 | 1.098 | 1.631 | 1.928 (LR) | 2.076 |
| 12 | 1.118 | 1.061 | 1.163 | 3.264 (LR) | 3.795 |

Nachylenie RelMAE względem log(h): +0.077 -> przewaga nad Naive NIE rośnie z h.

Podział B – LSTM-Δ: RelMAE (MAE / MAE Naive; < 1 = lepiej niż Naive)

| h | RelMAE LSTM | RelMAE SARIMA | RelMAE XGBoost | MAE najlepszego modelu bazowego | MAE LSTM |
|---|---|---|---|---|---|
| 1 | 0.991 | 1.013 | 1.413 | 0.615 (LR) | 0.700 |
| 3 | 0.956 | 1.029 | 1.159 | 1.603 (LR) | 1.768 |
| 6 | 1.004 | 1.084 | 1.094 | 2.364 (SVR) | 3.524 |
| 12 | 1.002 | 1.304 | 1.059 | 2.633 (SVR) | 5.365 |

Nachylenie RelMAE względem log(h): +0.008 -> przewaga nad Naive NIE rośnie z h.


## Kryteria sukcesu (sekcja 4.4)

**Podział A (LSTM-Δ)**
1. RelMAE < 1 i istotność DM vs Naive dla ≥1 horyzontu: **SPEŁNIONE**
3. Stabilność: średnie std MAE po ziarnach = 0.054, średnia |różnica| względem najlepszego modelu bazowego = 0.173 -> **SPEŁNIONE**

**Podział B (LSTM-Δ)**
1. RelMAE < 1 i istotność DM vs Naive dla ≥1 horyzontu: **NIESPEŁNIONE**
2. MAE (h=1) niższe niż XGBoost: 0.700 vs 0.997 -> **SPEŁNIONE**
3. Stabilność: średnie std MAE po ziarnach = 0.043, średnia |różnica| względem najlepszego modelu bazowego = 1.035 -> **SPEŁNIONE**
