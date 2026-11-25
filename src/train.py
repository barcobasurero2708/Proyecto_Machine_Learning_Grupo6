"""Pipeline de entrenamiento y evaluación de la entrega final.

Partición temporal (la misma de la entrega 1):

    enero      -> búsqueda de hiperparámetros: ajuste
    febrero    -> búsqueda de hiperparámetros: validación interna
    ene + feb  -> ENTRENAMIENTO de los modelos elegidos
    marzo      -> VALIDACIÓN: comparar familias y elegir el modelo final
    abril      -> TEST: se lee UNA sola vez, después de fijar el modelo

Uso (desde la raíz del proyecto):
    python main.py --etapa validacion --data-dir Data
    python main.py --etapa test --data-dir Data
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from src import evaluate as E
from src.data import load_population
from src.features import (ALL_FEATURES, CHARGE_FEATURES, TARGET, ClippedTargetRegressor, add_features,
                          build_preprocessor, quality_flags)
from src.model import linear_pipeline, make_features as linear_features

SEED = 42
TRAIN_MONTHS = ["2026-01", "2026-02"]
VAL_MONTH = "2026-03"
TEST_MONTH = "2026-04"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"

SEARCH_SPACE = {
    "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "lineal_absoluto": {"alpha": [1e-6, 1e-5, 1e-4, 1e-3]},
    "random_forest": {"max_depth": [12, 18, 24], "min_samples_leaf": [20, 50, 200],
                      "max_features": [0.3, 0.5]},
    "hgb_absoluto": {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31, 63, 127],
                     "min_samples_leaf": [50, 200, 1000], "l2_regularization": [0.0, 1.0, 10.0]},
    "hgb_cuadratico": {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [31, 63, 127],
                       "min_samples_leaf": [50, 200, 1000], "l2_regularization": [0.0, 1.0, 10.0]},
}
N_CONFIGS = {"ridge": None, "lineal_absoluto": None, "random_forest": 4, "hgb_absoluto": 6, "hgb_cuadratico": 4}
FAMILY = {"ridge": "lineal regularizado (pérdida cuadrática)",
          "lineal_absoluto": "lineal regularizado (pérdida absoluta)", "random_forest": "bagging de árboles",
          "hgb_absoluto": "gradient boosting (pérdida absoluta)",
          "hgb_cuadratico": "gradient boosting (pérdida cuadrática)"}


def make_model(name: str, params: dict, n_train: int, rf_rows: int, drop=()) -> ClippedTargetRegressor:
    if name == "ridge":
        prep, model = build_preprocessor(scale=True, seed=SEED, drop=drop), Ridge(random_state=SEED, **params)
    elif name == "lineal_absoluto":
        # epsilon_insensitive con epsilon=0 equivale a minimizar el error absoluto (regresión L1).
        prep = build_preprocessor(scale=True, seed=SEED, drop=drop)
        model = SGDRegressor(loss="epsilon_insensitive", epsilon=0.0, penalty="l2", max_iter=30,
                             tol=1e-4, learning_rate="invscaling", eta0=0.05, random_state=SEED, **params)
    elif name == "random_forest":
        prep = build_preprocessor(seed=SEED, drop=drop)
        model = RandomForestRegressor(n_estimators=60, max_samples=min(1.0, rf_rows / n_train),
                                      n_jobs=-1, random_state=SEED, **params)
    else:
        loss = "absolute_error" if name == "hgb_absoluto" else "squared_error"
        prep = build_preprocessor(seed=SEED, drop=drop)
        model = HistGradientBoostingRegressor(loss=loss, max_iter=500, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=20,
                                              random_state=SEED, **params)
    return ClippedTargetRegressor(Pipeline([("preprocesar", prep), ("modelo", model)]))


def predict_chunks(model, X: pd.DataFrame, chunk: int = 400_000) -> np.ndarray:
    return np.concatenate([model.predict(X.iloc[i:i + chunk]) for i in range(0, len(X), chunk)])


def features_chunks(df: pd.DataFrame, chunk: int = 500_000) -> pd.DataFrame:
    return pd.concat([add_features(df.iloc[i:i + chunk]) for i in range(0, len(df), chunk)])


def _json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- carga
def load_training(data_dir, months, n_per_month, seed=SEED):
    """Población completa de cada mes -> (muestra para modelos, muestra 5% de la
    entrega 1, todas las etiquetas, reporte de calidad, flujos, resumen)."""
    samples, linear_samples, ys, quality, flows, summary = [], [], [], [], [], []
    for month in months:
        print(f"  cargando {month}...", flush=True)
        pop, flow = load_population(data_dir, month)
        flows.append(flow)
        ys.append(pop[TARGET].to_numpy())
        flags = quality_flags(pop)
        quality.append(_quality_rows(month, flags))
        summary.append(_target_summary(month, pop[TARGET]))
        # Misma muestra 5% que la entrega 1 (misma semilla por mes y mismo orden de filas).
        linear_samples.append(pop.sample(frac=0.05, random_state=seed + int(month[-2:])))
        n = min(n_per_month, len(pop))
        samples.append(pop.sample(n=n, random_state=seed + 100 + int(month[-2:])))
        del pop, flags
        gc.collect()
    return (pd.concat(samples, ignore_index=True), pd.concat(linear_samples, ignore_index=True),
            np.concatenate(ys), quality, pd.concat(flows, ignore_index=True), summary)


def _quality_rows(month, flags: pd.DataFrame):
    rows = [{"mes": month, "regla": c, "n_marcados": int(flags[c].sum()),
             "pct": round(100 * float(flags[c].mean()), 3)} for c in flags.columns]
    any_flag = flags.any(axis=1)
    rows.append({"mes": month, "regla": "registro_sospechoso (alguna regla)",
                 "n_marcados": int(any_flag.sum()), "pct": round(100 * float(any_flag.mean()), 3)})
    return pd.DataFrame(rows)


def _target_summary(month, y: pd.Series) -> dict:
    return {"mes": month, "n": int(len(y)), "media": round(float(y.mean()), 3),
            "mediana": round(float(y.median()), 3), "p10": round(float(y.quantile(.1)), 3),
            "p90": round(float(y.quantile(.9)), 3), "p99": round(float(y.quantile(.99)), 3),
            "pct_cero": round(100 * float(y.eq(0).mean()), 3),
            "pct_mayor_100": round(100 * float(y.gt(100).mean()), 4)}


def fit_baselines(train_y, linear_sample):
    models = {}
    for strategy, name in [("mean", "baseline_media"), ("median", "baseline_mediana")]:
        models[name] = DummyRegressor(strategy=strategy).fit(np.zeros((len(train_y), 1)), train_y)
    lin = linear_pipeline().fit(linear_features(linear_sample), linear_sample[TARGET])
    models["baseline_lineal"] = lin
    return models


def predict_baselines(models, pop: pd.DataFrame) -> dict:
    n = len(pop)
    out = {name: models[name].predict(np.zeros((n, 1))) for name in ("baseline_media", "baseline_mediana")}
    lin = models["baseline_lineal"]
    out["baseline_lineal"] = np.maximum(np.concatenate(
        [lin.predict(linear_features(pop.iloc[i:i + 200_000])) for i in range(0, n, 200_000)]), 0)
    return out


# --------------------------------------------------------------------------- búsqueda
def hyperparameter_search(train: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    jan = train[train.mes == TRAIN_MONTHS[0]]
    feb = train[train.mes == TRAIN_MONTHS[1]]
    jan = jan.sample(n=min(args.n_busqueda, len(jan)), random_state=SEED)
    feb = feb.sample(n=min(args.n_busqueda, len(feb)), random_state=SEED)
    X_fit, y_fit = add_features(jan), jan[TARGET]
    X_val, y_val = add_features(feb), feb[TARGET].to_numpy()
    rows = []
    for name, space in SEARCH_SPACE.items():
        n_iter = N_CONFIGS[name]
        configs = (list(ParameterSampler(space, n_iter=n_iter, random_state=SEED)) if n_iter
                   else [dict(alpha=a) for a in space["alpha"]])
        for params in configs:
            t0 = time.time()
            model = make_model(name, params, len(X_fit), args.rf_filas).fit(X_fit, y_fit)
            mae = float(np.mean(np.abs(y_val - model.predict(X_val))))
            rows.append({"modelo": name, "familia": FAMILY[name], "params": json.dumps(params),
                         "MAE_pp": round(mae, 4), "seg": round(time.time() - t0, 1)})
            print(f"    {name:15s} {params} MAE_feb={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    best = {name: json.loads(t.loc[t.MAE_pp.idxmin(), "params"]) for name, t in table.groupby("modelo")}
    return table, best


# --------------------------------------------------------------------------- etapa 1
def stage_validation(args) -> dict:
    t_start = time.time()
    print("== VALIDACIÓN: entrenamiento ene-feb, validación marzo (abril no se lee) ==", flush=True)
    train, lin_sample, y_train_all, quality, flows, summary = load_training(
        args.data_dir, TRAIN_MONTHS, args.n_por_mes)
    val, val_flow = load_population(args.data_dir, VAL_MONTH)
    flows = pd.concat([flows, val_flow], ignore_index=True)
    flows.to_csv(OUT / "filtros_modelado.csv", index=False)
    quality.append(_quality_rows(VAL_MONTH, quality_flags(val)))
    pd.concat(quality, ignore_index=True).to_csv(OUT / "limpieza_reporte.csv", index=False)
    summary.append(_target_summary(VAL_MONTH, val[TARGET]))
    drift = pd.DataFrame(summary)
    drift.to_csv(OUT / "deriva_mensual.csv", index=False)
    rng = np.random.default_rng(SEED)
    def _sub(y):
        return rng.choice(y, min(200_000, len(y)), replace=False)
    hist = {m: _sub(train.loc[train.mes == m, TARGET].to_numpy()) for m in TRAIN_MONTHS}
    hist[VAL_MONTH] = _sub(val[TARGET].to_numpy())
    E.plot_monthly_target(drift, hist, FIG / "09_deriva_objetivo_por_mes.png")
    print(f"  train muestra={len(train):,} | etiquetas ene-feb={len(y_train_all):,} | val={len(val):,}")

    print("  búsqueda de hiperparámetros (enero -> febrero)...", flush=True)
    search, best_params = hyperparameter_search(train, args)
    search.to_csv(OUT / "busqueda_hiperparametros.csv", index=False)
    E.plot_search(search, FIG / "10_busqueda_hiperparametros.png")

    X_train = add_features(train)
    y_train = train[TARGET]
    X_val = features_chunks(val)
    y_val = val[TARGET].to_numpy()
    fare_val = val["fare_amount"].to_numpy()

    baselines = fit_baselines(y_train_all, lin_sample)
    preds = predict_baselines(baselines, val)
    results = {n: E.regression_metrics(y_val, p, fare_val) | {"n_entrenamiento": int(
        len(lin_sample) if n == "baseline_lineal" else len(y_train_all))} for n, p in preds.items()}
    fitted = {}
    for name, params in best_params.items():
        t0 = time.time()
        model = make_model(name, params, len(X_train), args.rf_filas).fit(X_train, y_train)
        fit_s = time.time() - t0
        preds[name] = predict_chunks(model, X_val)
        results[name] = E.regression_metrics(y_val, preds[name], fare_val) | {
            "n_entrenamiento": int(len(X_train)), "fit_seg": round(fit_s, 1), "params": params}
        fitted[name] = model
        print(f"    {name:15s} MAE_mar={results[name]['MAE_pp']:.4f} ({fit_s:.0f}s)", flush=True)
    for name, r in results.items():
        print(f"  {name:18s} MAE={r['MAE_pp']:.4f} RMSE={r['RMSE_pp']:.3f} RMSE(>=1USD)={r['RMSE_pp_tarifa_ge_1']:.3f}")

    # Ablación: el mejor tipo de modelo sin las variables de cobro previas a la propina
    # (mismas variables que la entrega 1 + derivadas). Mide cuánto aporta el subtotal.
    order = ["baseline_media", "baseline_mediana", "baseline_lineal", "ridge", "lineal_absoluto",
             "random_forest", "hgb_absoluto", "hgb_cuadratico"]
    best_tmp = min([k for k in order if not k.startswith("baseline")], key=lambda k: results[k]["MAE_pp"])
    ablation_model = make_model(best_tmp, best_params[best_tmp], len(X_train), args.rf_filas,
                                drop=CHARGE_FEATURES).fit(X_train, y_train)
    ablation = E.regression_metrics(y_val, predict_chunks(ablation_model, X_val), fare_val)
    ablation |= {"modelo_base": best_tmp, "variables_quitadas": CHARGE_FEATURES}
    del ablation_model
    print(f"  ablación sin recargos ({best_tmp}): MAE={ablation['MAE_pp']:.4f}")
    results = {k: results[k] for k in order}
    candidates = [k for k in results if not k.startswith("baseline")]
    best = min(candidates, key=lambda k: results[k]["MAE_pp"])
    base = results["baseline_lineal"]["MAE_pp"]
    E.plot_model_comparison(results, FIG / "11_comparacion_modelos_validacion.png",
                            "11. Comparación de modelos | validación en todo marzo",
                            f"{len(y_val):,} viajes; mismos datos para todos los modelos")
    E.plot_pred_vs_real(y_val, preds[best], FIG / "12_pred_vs_real_validacion.png",
                        f"12. {best} | validación (marzo)")

    # ---- interpretación (muestra de validación)
    model = fitted[best]
    idx = np.random.default_rng(SEED).choice(len(X_val), args.n_importancia, replace=False)
    Xs, ys = X_val.iloc[idx], y_val[idx]
    pi = permutation_importance(model, Xs, ys, scoring="neg_mean_absolute_error", n_repeats=3,
                                random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": ALL_FEATURES, "importancia_media": pi.importances_mean.round(4),
                        "desv": pi.importances_std.round(4)}).sort_values("importancia_media", ascending=False)
    imp.to_csv(OUT / "importancia_variables.csv", index=False)
    E.plot_importance(imp, FIG / "13_importancia_permutacion.png",
                      f"13. Importancia por permutación | {best}, {args.n_importancia:,} viajes de marzo")
    pdp, pdp_rows = {}, []
    Xp = Xs.iloc[:5000]
    for feat, (lo, hi) in {"ratio_subtotal_fare": (1.0, 2.5), "fare_amount": (3, 80),
                           "duration_min": (2, 90)}.items():
        grid = np.linspace(lo, hi, 30)
        values = []
        for g in grid:
            Xg = Xp.copy()
            Xg[feat] = g
            values.append(float(model.predict(Xg).mean()))
        pdp[feat] = (grid, values)
        pdp_rows += [{"variable": feat, "valor": round(float(g), 3), "prediccion_media": round(v, 3)}
                     for g, v in zip(grid, values)]
    pd.DataFrame(pdp_rows).to_csv(OUT / "dependencia_parcial.csv", index=False)
    E.plot_partial_dependence(pdp, FIG / "14_dependencia_parcial.png")

    # ---- análisis de errores (todo marzo)
    seg = E.segment_errors(val, preds[best], baseline_pred=preds["baseline_lineal"])
    seg.to_csv(OUT / "errores_por_segmento_validacion.csv", index=False)
    E.zone_errors(val, preds[best]).to_csv(OUT / "errores_por_zona_validacion.csv", index=False)
    E.plot_segment(seg, "tarifa base (USD)", FIG / "15_mae_por_tarifa.png",
                   "15. MAE por tarifa base | validación (marzo)")
    E.plot_error_by_hour(val, preds[best], FIG / "16_error_por_hora.png")
    E.plot_zero_tip(seg, FIG / "17_error_propina_cero.png")
    worst = val.assign(prediccion=preds[best], error_abs=np.abs(y_val - preds[best]))
    cols = ["tpep_pickup_datetime", "fare_amount", "trip_distance", "duration_min", "PULocationID",
            "DOLocationID", "RatecodeID", "VendorID", TARGET, "prediccion", "error_abs"]
    worst.nlargest(25, "error_abs")[cols].round(3).to_csv(OUT / "casos_mayor_error_validacion.csv", index=False)
    del worst

    out = {"etapa": "validacion", "split": {"busqueda_ajuste": TRAIN_MONTHS[0],
                                             "busqueda_validacion": TRAIN_MONTHS[1],
                                             "entrenamiento": TRAIN_MONTHS, "validacion": VAL_MONTH,
                                             "test_reservado": TEST_MONTH, "abril_leido": False},
           "n_entrenamiento_modelos": int(len(X_train)), "n_etiquetas_ene_feb": int(len(y_train_all)),
           "n_validacion": int(len(y_val)), "muestreo": {"n_por_mes": args.n_por_mes,
                                                           "n_busqueda_por_mes": args.n_busqueda,
                                                           "rf_filas_por_arbol": args.rf_filas},
           "unidad": "puntos porcentuales (pp) de la tarifa base",
           "recorte_objetivo_entrenamiento": "cuantil 0.999 del entrenamiento (solo al ajustar)",
           "resultados": results, "mejor_modelo": best, "parametros_elegidos": best_params,
           "ablacion_sin_variables_de_cobro": ablation,
           "mejora_mae_vs_baseline_lineal_pct": round(100 * (1 - results[best]["MAE_pp"] / base), 2),
           "mejora_mae_vs_baseline_mediana_pct": round(
               100 * (1 - results[best]["MAE_pp"] / results["baseline_mediana"]["MAE_pp"]), 2),
           "features": ALL_FEATURES, "seed": SEED,
           "duracion_seg": round(time.time() - t_start)}
    _json(OUT / "metrics_validacion.json", out)
    print(f"Validación lista en {time.time()-t_start:.0f}s. Mejor: {best}")
    return out


# --------------------------------------------------------------------------- etapa 2
def stage_test(args) -> dict:
    t_start = time.time()
    path = OUT / "metrics_validacion.json"
    if not path.exists():
        raise SystemExit("Primero ejecuta la etapa de validación.")
    val_info = json.loads(path.read_text(encoding="utf-8"))
    best, params = val_info["mejor_modelo"], val_info["parametros_elegidos"][val_info["mejor_modelo"]]
    print(f"== TEST: entrenamiento ene-mar, test abril | modelo fijado en validación: {best} {params} ==")
    months = TRAIN_MONTHS + [VAL_MONTH]
    train, lin_sample, y_all, _, _, _ = load_training(args.data_dir, months, args.n_por_mes)
    X_train = add_features(train)
    baselines = fit_baselines(y_all, lin_sample)
    model = make_model(best, params, len(X_train), args.rf_filas).fit(X_train, train[TARGET])
    n_fit = int(len(X_train))
    del train, X_train
    gc.collect()

    test, test_flow = load_population(args.data_dir, TEST_MONTH)
    test_flow.to_csv(OUT / "filtros_test.csv", index=False)
    y_test, fare_test = test[TARGET].to_numpy(), test["fare_amount"].to_numpy()
    preds = predict_baselines(baselines, test)
    preds[best] = predict_chunks(model, features_chunks(test))
    results = {n: E.regression_metrics(y_test, p, fare_test) for n, p in preds.items()}
    for n, r in results.items():
        print(f"  {n:18s} MAE={r['MAE_pp']:.4f} RMSE={r['RMSE_pp']:.3f}")
    seg = E.segment_errors(test, preds[best], baseline_pred=preds["baseline_lineal"])
    seg.to_csv(OUT / "errores_por_segmento_test.csv", index=False)
    E.plot_model_comparison(results, FIG / "18_comparacion_test.png",
                            "18. Evaluación final única | test en todo abril",
                            f"{len(y_test):,} viajes; modelos reentrenados con enero-marzo")
    E.plot_pred_vs_real(y_test, preds[best], FIG / "19_pred_vs_real_test.png", f"19. {best} | test (abril)")
    try:
        joblib.dump(model, OUT / "modelo_final.joblib", compress=3)
    except Exception as exc:  # regenerable; no es crítico
        print("No se pudo guardar el modelo:", exc)
    base = results["baseline_lineal"]["MAE_pp"]
    out = {"etapa": "test", "split": {"entrenamiento": months, "test": TEST_MONTH},
           "n_entrenamiento_modelo": n_fit,
           "n_etiquetas_ene_mar": int(len(y_all)), "n_test": int(len(y_test)),
           "modelo_final": best, "parametros": params, "resultados": results,
           "mejora_mae_vs_baseline_lineal_pct": round(100 * (1 - results[best]["MAE_pp"] / base), 2),
           "mejora_mae_vs_baseline_mediana_pct": round(
               100 * (1 - results[best]["MAE_pp"] / results["baseline_mediana"]["MAE_pp"]), 2),
           "resumen_objetivo_test": _target_summary(TEST_MONTH, test[TARGET]),
           "seed": SEED, "duracion_seg": round(time.time() - t_start)}
    _json(OUT / "metrics_test.json", out)
    print(f"Test listo en {time.time()-t_start:.0f}s")
    return out
