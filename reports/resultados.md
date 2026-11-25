# Resultados generados por `main.py`

_Archivo generado automáticamente. No editar a mano: el informe final cita estas tablas._

## Validación: entrenamiento 2026-01, 2026-02 → marzo
Viajes de validación: 2,565,402. Métricas en puntos porcentuales (pp).

| modelo | n | MAE_pp | RMSE_pp | R2 | MedAE_pp | P90_AE_pp | RMSE_pp_tarifa_ge_1 | n_entrenamiento | fit_seg |
|---|---|---|---|---|---|---|---|---|---|
| baseline_media | 2565402.0 | 9.7479 | 107.6096 | -6e-06 | 6.6268 | 25.3567 | 25.3268 | 4225076.0 |  |
| baseline_mediana | 2565402.0 | 9.6848 | 107.6168 | -0.00014 | 5.9898 | 26.3758 | 25.3619 | 4225076.0 |  |
| baseline_lineal | 2565402.0 | 7.5237 | 107.2835 | 0.006047 | 4.3274 | 17.3505 | 24.0122 | 211254.0 |  |
| ridge | 2565402 | 7.4818 | 107.1408 | 0.008688 | 3.9915 | 17.0808 | 23.9262 | 800000 | 6.3 |
| lineal_absoluto | 2565402 | 6.532 | 105.7836 | 0.033644 | 0.9771 | 18.9481 | 24.0777 | 800000 | 17.5 |
| random_forest | 2565402 | 7.4227 | 107.1764 | 0.00803 | 3.9196 | 16.9289 | 23.6321 | 800000 | 26.4 |
| hgb_absoluto | 2565402 | 6.3139 | 107.2321 | 0.006999 | 0.1031 | 19.1853 | 23.9557 | 800000 | 94.8 |
| hgb_cuadratico | 2565402 | 7.4126 | 107.1535 | 0.008453 | 3.9245 | 16.9051 | 23.5159 | 800000 | 42.8 |

**Modelo elegido por MAE:** `hgb_absoluto` {"min_samples_leaf": 50, "max_leaf_nodes": 31, "learning_rate": 0.2, "l2_regularization": 0.0}. Mejora de MAE frente al baseline lineal: **16.08%**; frente a la mediana: **34.81%**.

## Búsqueda de hiperparámetros (ajuste en enero, validación en febrero)

| modelo | familia | params | MAE_pp | seg |
|---|---|---|---|---|
| ridge | lineal regularizado (pérdida cuadrática) | {"alpha": 0.1} | 7.5486 | 2.4 |
| ridge | lineal regularizado (pérdida cuadrática) | {"alpha": 1.0} | 7.5487 | 2.4 |
| ridge | lineal regularizado (pérdida cuadrática) | {"alpha": 10.0} | 7.5492 | 1.9 |
| ridge | lineal regularizado (pérdida cuadrática) | {"alpha": 100.0} | 7.5501 | 2.0 |
| ridge | lineal regularizado (pérdida cuadrática) | {"alpha": 1000.0} | 7.5486 | 2.2 |
| lineal_absoluto | lineal regularizado (pérdida absoluta) | {"alpha": 1e-06} | 6.6432 | 5.5 |
| lineal_absoluto | lineal regularizado (pérdida absoluta) | {"alpha": 1e-05} | 6.6438 | 6.0 |
| lineal_absoluto | lineal regularizado (pérdida absoluta) | {"alpha": 0.0001} | 6.6427 | 5.5 |
| lineal_absoluto | lineal regularizado (pérdida absoluta) | {"alpha": 0.001} | 6.6669 | 5.8 |
| random_forest | bagging de árboles | {"min_samples_leaf": 20, "max_features": 0.3, "max_depth": 12} | 7.4667 | 17.8 |
| random_forest | bagging de árboles | {"min_samples_leaf": 50, "max_features": 0.3, "max_depth": 12} | 7.467 | 17.1 |
| random_forest | bagging de árboles | {"min_samples_leaf": 200, "max_features": 0.3, "max_depth": 18} | 7.4724 | 16.4 |
| random_forest | bagging de árboles | {"min_samples_leaf": 200, "max_features": 0.5, "max_depth": 12} | 7.4675 | 22.9 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 50, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 1.0} | 6.4191 | 49.4 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 50, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 0.0} | 6.4476 | 43.4 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 200, "max_leaf_nodes": 63, "learning_rate": 0.2, "l2_regularization": 0.0} | 6.4309 | 32.2 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 200, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 1.0} | 6.4259 | 45.9 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 50, "max_leaf_nodes": 31, "learning_rate": 0.2, "l2_regularization": 0.0} | 6.4145 | 37.5 |
| hgb_absoluto | gradient boosting (pérdida absoluta) | {"min_samples_leaf": 200, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 1.0} | 6.4397 | 44.2 |
| hgb_cuadratico | gradient boosting (pérdida cuadrática) | {"min_samples_leaf": 50, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 1.0} | 7.4786 | 11.7 |
| hgb_cuadratico | gradient boosting (pérdida cuadrática) | {"min_samples_leaf": 50, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 0.0} | 7.4721 | 14.2 |
| hgb_cuadratico | gradient boosting (pérdida cuadrática) | {"min_samples_leaf": 200, "max_leaf_nodes": 63, "learning_rate": 0.2, "l2_regularization": 0.0} | 7.4783 | 4.8 |
| hgb_cuadratico | gradient boosting (pérdida cuadrática) | {"min_samples_leaf": 200, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 1.0} | 7.4802 | 10.5 |

## Registros sospechosos marcados (no se eliminan)

| mes | regla | n_marcados | pct |
|---|---|---|---|
| 2026-01 | duracion_menor_1_min | 17237 | 0.78 |
| 2026-01 | duracion_mayor_180_min | 1425 | 0.064 |
| 2026-01 | distancia_cero_o_faltante | 17268 | 0.781 |
| 2026-01 | distancia_mayor_100_mi | 17 | 0.001 |
| 2026-01 | velocidad_mayor_80_mph | 287 | 0.013 |
| 2026-01 | registro_sospechoso (alguna regla) | 23805 | 1.077 |
| 2026-02 | duracion_menor_1_min | 14657 | 0.727 |
| 2026-02 | duracion_mayor_180_min | 1172 | 0.058 |
| 2026-02 | distancia_cero_o_faltante | 15527 | 0.771 |
| 2026-02 | distancia_mayor_100_mi | 25 | 0.001 |
| 2026-02 | velocidad_mayor_80_mph | 280 | 0.014 |
| 2026-02 | registro_sospechoso (alguna regla) | 21037 | 1.044 |
| 2026-03 | duracion_menor_1_min | 19174 | 0.747 |
| 2026-03 | duracion_mayor_180_min | 753 | 0.029 |
| 2026-03 | distancia_cero_o_faltante | 19400 | 0.756 |
| 2026-03 | distancia_mayor_100_mi | 22 | 0.001 |
| 2026-03 | velocidad_mayor_80_mph | 466 | 0.018 |
| 2026-03 | registro_sospechoso (alguna regla) | 26241 | 1.023 |

## Objetivo por mes (poblaciones completas)

| mes | n | media | mediana | p10 | p90 | p99 | pct_cero | pct_mayor_100 |
|---|---|---|---|---|---|---|---|---|
| 2026-01 | 2210259 | 25.472 | 26.566 | 3.731 | 38.056 | 57.083 | 9.489 | 0.186 |
| 2026-02 | 2014817 | 25.231 | 26.25 | 3.333 | 37.692 | 56.207 | 9.556 | 0.1745 |
| 2026-03 | 2565402 | 25.103 | 26.421 | 0.165 | 37.742 | 56.462 | 9.891 | 0.169 |

## Importancia por permutación

| feature | importancia_media | desv |
|---|---|---|
| ratio_subtotal_fare | 3.1651 | 0.0275 |
| RatecodeID_cat | 1.489 | 0.0013 |
| fare_amount | 0.5046 | 0.0035 |
| VendorID_cat | 0.4508 | 0.0086 |
| subtotal_pre_tip | 0.1906 | 0.0011 |
| extra | 0.0987 | 0.0008 |
| Airport_fee | 0.0349 | 0.0005 |
| improvement_surcharge | 0.0269 | 0.0004 |
| cbd_congestion_fee | 0.0237 | 0.0005 |
| hour_cos | 0.0234 | 0.0002 |
| hour_sin | 0.0158 | 0.0003 |
| tolls_amount | 0.015 | 0.0002 |
| duration_min | 0.0098 | 0.0005 |
| trip_distance | 0.0068 | 0.0002 |
| is_weekend | 0.005 | 0.0001 |
| route | 0.0037 | 0.0001 |
| DOLocationID | 0.0037 | 0.0001 |
| congestion_surcharge | 0.003 | 0.0001 |
| do_airport | 0.0019 | 0.0 |
| PULocationID | 0.0018 | 0.0001 |
| mta_tax | 0.0008 | 0.0001 |
| pickup_hour | 0.0006 | 0.0002 |
| passenger_count | 0.0003 | 0.0 |
| unknown_zone | 0.0001 | 0.0 |
| log_fare | 0.0 | 0.0 |
| speed_mph | 0.0 | 0.0 |
| is_holiday | 0.0 | 0.0 |
| pu_airport | 0.0 | 0.0 |
| ratecode_missing | 0.0 | 0.0 |
| same_zone | 0.0 | 0.0 |
| pickup_weekday | 0.0 | 0.0 |
| registro_sospechoso | 0.0 | 0.0 |

## Errores por segmento (validación)

| valor | segmento | n | pct_viajes | mediana_real_pp | MAE_pp | MedAE_pp | sesgo_medio_pp | MAE_baseline_pp | mejora_pct |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | tarifa base (USD) | 18 | 0.0 | 80.68 | 15958.327 | 96.488 | -15885.376 | 15959.899 | 0.0 |
| 1-5 | tarifa base (USD) | 32262 | 1.26 | 45.45 | 30.45 | 7.275 | -14.2 | 33.672 | 9.6 |
| 5-10 | tarifa base (USD) | 807850 | 31.49 | 32.26 | 7.263 | 0.097 | 2.923 | 8.356 | 13.1 |
| 10-20 | tarifa base (USD) | 982082 | 38.28 | 26.76 | 5.803 | 0.09 | 3.09 | 6.797 | 14.6 |
| 20-50 | tarifa base (USD) | 560467 | 21.85 | 23.14 | 4.425 | 0.069 | 2.32 | 6.002 | 26.3 |
| 50+ | tarifa base (USD) | 182723 | 7.12 | 21.36 | 4.82 | 0.233 | 2.946 | 6.227 | 22.6 |
| 0-1 | distancia (mi) | 643547 | 25.09 | 32.37 | 9.012 | 0.161 | 1.601 | 10.381 | 13.2 |
| 1-2 | distancia (mi) | 824715 | 32.15 | 28.19 | 6.064 | 0.075 | 3.068 | 7.031 | 13.8 |
| 2-5 | distancia (mi) | 640661 | 24.97 | 24.97 | 5.416 | 0.081 | 2.958 | 6.665 | 18.7 |
| 5-10 | distancia (mi) | 229633 | 8.95 | 22.78 | 4.409 | 0.124 | 2.294 | 5.9 | 25.3 |
| 10-20 | distancia (mi) | 200602 | 7.82 | 21.12 | 3.942 | 0.137 | 2.13 | 5.195 | 24.1 |
| 20+ | distancia (mi) | 26244 | 1.02 | 20.27 | 4.696 | 0.198 | 3.077 | 5.908 | 20.5 |
| madrugada (0-5) | franja horaria | 180402 | 7.03 | 26.25 | 8.774 | 0.256 | 3.534 | 9.958 | 11.9 |
| mañana pico (6-9) | franja horaria | 316785 | 12.35 | 25.24 | 5.743 | 0.103 | 2.324 | 6.678 | 14.0 |
| día (10-15) | franja horaria | 839671 | 32.73 | 25.59 | 5.582 | 0.045 | 1.736 | 6.716 | 16.9 |
| tarde pico (16-19) | franja horaria | 702163 | 27.37 | 27.88 | 6.702 | 0.12 | 3.059 | 8.124 | 17.5 |
| noche (20-23) | franja horaria | 526381 | 20.52 | 27.06 | 6.465 | 0.133 | 2.871 | 7.687 | 15.9 |
| no | fin de semana | 1890978 | 73.71 | 26.51 | 6.169 | 0.093 | 2.419 | 7.394 | 16.6 |
| sí | fin de semana | 674424 | 26.29 | 26.25 | 6.721 | 0.139 | 2.84 | 7.888 | 14.8 |
| desde aeropuerto | aeropuerto | 191253 | 7.46 | 23.36 | 5.465 | 0.539 | 2.61 | 6.971 | 21.6 |
| hacia aeropuerto | aeropuerto | 46919 | 1.83 | 23.84 | 5.276 | 0.297 | 2.409 | 6.477 | 18.5 |
| sin aeropuerto | aeropuerto | 2327230 | 90.72 | 26.84 | 6.405 | 0.067 | 2.526 | 7.59 | 15.6 |
| 1 | VendorID | 580676 | 22.63 | 25.32 | 5.307 | 0.097 | 0.873 | 6.449 | 17.7 |
| 2 | VendorID | 1984726 | 77.37 | 26.69 | 6.609 | 0.107 | 3.015 | 7.838 | 15.7 |
| no | RatecodeID faltante | 2565402 | 100.0 | 26.42 | 6.314 | 0.103 | 2.53 | 7.524 | 16.1 |
| no | registro sospechoso | 2539161 | 98.98 | 26.5 | 5.968 | 0.098 | 2.811 | 7.164 | 16.7 |
| sí | registro sospechoso | 26241 | 1.02 | 20.82 | 39.812 | 5.859 | -24.644 | 42.32 | 5.9 |
| cero | propina real | 253733 | 9.89 | 0.0 | 13.115 | 0.0 | 13.115 | 12.508 | -4.9 |
| positiva | propina real | 2311669 | 90.11 | 27.42 | 5.567 | 0.111 | 1.368 | 6.977 | 20.2 |

## Test único: entrenamiento ene-mar → abril (2,591,610 viajes)

| modelo | n | MAE_pp | RMSE_pp | R2 | MedAE_pp | P90_AE_pp | RMSE_pp_tarifa_ge_1 |
|---|---|---|---|---|---|---|---|
| baseline_media | 2591610.0 | 10.1089 | 654.9891 | -0.0 | 6.3777 | 25.2607 | 24.9242 |
| baseline_mediana | 2591610.0 | 10.0539 | 654.99 | -3e-06 | 5.9188 | 26.3889 | 24.9729 |
| baseline_lineal | 2591610.0 | 7.9684 | 654.9165 | 0.000222 | 4.3949 | 17.3223 | 23.6253 |
| hgb_absoluto | 2591610.0 | 6.7061 | 654.9238 | 0.0002 | 0.1516 | 19.0328 | 23.6846 |

Mejora de MAE del modelo final frente al baseline lineal en test: **15.84%**; frente a la mediana: **33.3%**.

## Errores por segmento (test)

| valor | segmento | n | pct_viajes | mediana_real_pp | MAE_pp | MedAE_pp | sesgo_medio_pp | MAE_baseline_pp | mejora_pct |
|---|---|---|---|---|---|---|---|---|---|
| 0-1 | tarifa base (USD) | 23 | 0.0 | 2657.89 | 66022.205 | 2614.217 | -65993.666 | 66024.056 | 0.0 |
| 1-5 | tarifa base (USD) | 30857 | 1.19 | 45.45 | 29.756 | 7.064 | -14.292 | 32.562 | 8.6 |
| 5-10 | tarifa base (USD) | 768873 | 29.67 | 32.26 | 7.217 | 0.082 | 2.841 | 8.258 | 12.6 |
| 10-20 | tarifa base (USD) | 1001204 | 38.63 | 26.69 | 5.773 | 0.127 | 3.04 | 6.801 | 15.1 |
| 20-50 | tarifa base (USD) | 610686 | 23.56 | 23.14 | 4.489 | 0.153 | 2.335 | 6.291 | 28.6 |
| 50+ | tarifa base (USD) | 179967 | 6.94 | 21.36 | 4.846 | 0.326 | 2.922 | 6.263 | 22.6 |
| 0-1 | distancia (mi) | 648500 | 25.02 | 32.31 | 9.181 | 0.147 | 1.242 | 10.463 | 12.3 |
| 1-2 | distancia (mi) | 830936 | 32.06 | 27.85 | 5.994 | 0.102 | 3.024 | 6.977 | 14.1 |
| 2-5 | distancia (mi) | 649020 | 25.04 | 24.8 | 6.895 | 0.152 | 1.345 | 8.3 | 16.9 |
| 5-10 | distancia (mi) | 238731 | 9.21 | 22.75 | 4.458 | 0.217 | 2.272 | 6.176 | 27.8 |
| 10-20 | distancia (mi) | 198446 | 7.66 | 21.09 | 3.96 | 0.155 | 2.059 | 5.328 | 25.7 |
| 20+ | distancia (mi) | 25977 | 1.0 | 20.29 | 4.639 | 0.259 | 3.018 | 5.791 | 19.9 |
| madrugada (0-5) | franja horaria | 161644 | 6.24 | 26.25 | 10.323 | 0.307 | 2.0 | 11.561 | 10.7 |
| mañana pico (6-9) | franja horaria | 322661 | 12.45 | 25.16 | 5.57 | 0.135 | 2.321 | 6.565 | 15.2 |
| día (10-15) | franja horaria | 857022 | 33.07 | 25.59 | 6.775 | 0.095 | 0.434 | 7.978 | 15.1 |
| tarde pico (16-19) | franja horaria | 706000 | 27.24 | 27.65 | 6.525 | 0.185 | 3.098 | 8.013 | 18.6 |
| noche (20-23) | franja horaria | 544283 | 21.0 | 26.76 | 6.432 | 0.189 | 2.92 | 7.66 | 16.0 |
| no | fin de semana | 1958084 | 75.55 | 26.38 | 6.218 | 0.145 | 2.234 | 7.494 | 17.0 |
| sí | fin de semana | 633526 | 24.45 | 26.09 | 8.215 | 0.18 | 1.335 | 9.436 | 12.9 |
| desde aeropuerto | aeropuerto | 189608 | 7.32 | 23.49 | 5.565 | 0.664 | 2.523 | 7.171 | 22.4 |
| hacia aeropuerto | aeropuerto | 47607 | 1.84 | 23.96 | 5.255 | 0.374 | 2.207 | 6.539 | 19.6 |
| sin aeropuerto | aeropuerto | 2354395 | 90.85 | 26.69 | 6.827 | 0.114 | 1.969 | 8.062 | 15.3 |
| 1 | VendorID | 588261 | 22.7 | 25.08 | 7.304 | 0.125 | -1.242 | 8.49 | 14.0 |
| 2 | VendorID | 2003349 | 77.3 | 26.5 | 6.531 | 0.155 | 2.97 | 7.815 | 16.4 |
| no | RatecodeID faltante | 2591610 | 100.0 | 26.25 | 6.706 | 0.152 | 2.014 | 7.968 | 15.8 |
| no | registro sospechoso | 2565151 | 98.98 | 26.33 | 6.289 | 0.15 | 2.368 | 7.544 | 16.6 |
| sí | registro sospechoso | 26459 | 1.02 | 20.74 | 47.143 | 5.469 | -32.29 | 49.153 | 4.1 |
| cero | propina real | 260734 | 10.06 | 0.0 | 12.791 | 0.0 | 12.791 | 12.389 | -3.2 |
| positiva | propina real | 2330876 | 89.94 | 27.17 | 6.026 | 0.154 | 0.809 | 7.474 | 19.4 |
