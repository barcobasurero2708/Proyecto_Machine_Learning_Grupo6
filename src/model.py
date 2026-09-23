"""Baselines de media, mediana y regresión lineal sin propina ni total en X."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC = ["fare_amount", "trip_distance", "duration_min", "passenger_count"]
CATEGORICAL = ["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "pickup_hour", "pickup_weekday"]
FEATURES = NUMERIC + CATEGORICAL


def make_features(df):
    """Lista permitida: ninguna columna ajena a FEATURES puede pasar a X."""
    x = df[FEATURES].copy()
    x[NUMERIC] = x[NUMERIC].replace([np.inf, -np.inf], np.nan)
    x.loc[x.trip_distance.lt(0), "trip_distance"] = np.nan
    # Recuento cero se conserva; no se supone que signifique un dato faltante.
    x.loc[x.passenger_count.lt(0), "passenger_count"] = np.nan
    for column in CATEGORICAL:
        x[column] = x[column].fillna(-1).astype(str)
    return x


def linear_pipeline():
    numeric = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler(with_mean=False)),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=100, sparse_output=True)
    preprocessing = ColumnTransformer([
        ("numericas", numeric, NUMERIC), ("categoricas", categorical, CATEGORICAL)
    ])
    return Pipeline([("preprocesar", preprocessing), ("regresion", LinearRegression())])


def metrics(y, prediction):
    return {"n_evaluacion": len(y), "MAE_pp": float(mean_absolute_error(y, prediction)),
            "RMSE_pp": float(np.sqrt(mean_squared_error(y, prediction))),
            "R2": float(r2_score(y, prediction))}


def run_baselines(result, root):
    y_train = np.concatenate([result["targets"]["2026-01"], result["targets"]["2026-02"]])
    validation = result["validacion"]
    y_valid = validation.tip_percentage.to_numpy()
    train = result["muestra"].loc[result["muestra"].mes.isin(["2026-01", "2026-02"])].copy()
    rows, sensitivity, predictions = [], [], {}
    constants = {}
    for strategy, name in [("mean", "Media"), ("median", "Mediana")]:
        dummy = DummyRegressor(strategy=strategy)
        dummy.fit(np.zeros((len(y_train), 1), dtype=np.uint8), y_train)
        p = dummy.predict(np.zeros((len(y_valid), 1), dtype=np.uint8))
        constants[name] = float(dummy.constant_[0, 0])
        rows.append({"modelo": name, "n_entrenamiento": len(y_train), **metrics(y_valid, p)})
        predictions[name] = p
    print(f"Ajustando regresión lineal en {len(train):,} viajes de entrenamiento...", flush=True)
    model = linear_pipeline()
    model.fit(make_features(train), train.tip_percentage)
    prediction = np.empty(len(validation))
    for begin in range(0, len(validation), 100_000):
        end = min(begin + 100_000, len(validation))
        prediction[begin:end] = model.predict(make_features(validation.iloc[begin:end]))
    # Restricción física previa: una propina predicha no puede ser negativa.
    negative = float((prediction < 0).mean())
    prediction = np.maximum(prediction, 0)
    rows.append({"modelo": "Lineal_log1p", "n_entrenamiento": len(train), **metrics(y_valid, prediction)})
    predictions["Lineal_log1p"] = prediction

    # Diagnóstico adicional. No sustituye la evaluación principal ni cambia y.
    regular_fare = validation.fare_amount.ge(1).to_numpy()
    for name, p in predictions.items():
        sensitivity.append({"modelo": name, "segmento": "tarifa_mayor_igual_1_USD",
                            **metrics(y_valid[regular_fare], p[regular_fare])})
    result["metricas"] = pd.DataFrame(rows)
    result["sensibilidad"] = pd.DataFrame(sensitivity)
    output = Path(root) / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    result["metricas"].to_csv(output / "baseline_metrics.csv", index=False)
    result["sensibilidad"].to_csv(output / "sensibilidad_tarifa.csv", index=False)
    payload = {"seed": result["seed"], "sample_frac_lineal": result["sample_frac"],
               "entrenamiento": ["2026-01", "2026-02"], "validacion": "2026-03",
               "test_reservado": "2026-04", "abril_evaluado": False,
               "unidad_error": "puntos porcentuales de la tarifa base",
               "constantes": constants, "metricas": rows,
               "sensibilidad_no_principal": sensitivity,
               "lineal_fraccion_predicciones_negativas_antes_recorte": negative,
               "features": FEATURES}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
