"""Variables y preprocesamiento de la entrega final (scikit-learn Pipelines).

Instante de predicción (igual que en la entrega 1): cierre del taxímetro, con el
pasajero que ya eligió tarjeta y antes de introducir la propina.

Control de leakage
------------------
- Lista permitida explícita (`ALL_FEATURES`). Ninguna otra columna llega a X.
- Prohibidas: `tip_amount`, `total_amount`, `tip_percentage`, `payment_type` y
  cualquier reconstrucción que use la propina.
- Nuevas en la entrega final: los recargos (`extra`, `mta_tax`, `tolls_amount`,
  `improvement_surcharge`, `congestion_surcharge`, `Airport_fee`,
  `cbd_congestion_fee`). La propuesta los dejó fuera "hasta confirmar que se
  determinan antes de registrar la propina". El diccionario de la TLC los
  describe como cargos del taxímetro o de la tarifa, y en enero la mediana de
  `tip_amount / (tarifa + recargos)` es exactamente 20%, lo que indica que la
  pantalla de propina se calcula sobre ese subtotal ya conocido. Se suman SIN
  `total_amount`, por lo que el subtotal no contiene la propina.
- Lo que depende de los datos (imputación, escalado, categorías frecuentes y
  target encoding) se ajusta dentro del Pipeline, solo con entrenamiento. El
  TargetEncoder usa cross-fitting de 5 particiones al transformar el propio
  entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

TARGET = "tip_percentage"
AIRPORT_ZONES = (1, 132, 138)      # EWR, JFK, LaGuardia según el lookup de la TLC
UNKNOWN_ZONES = (264, 265)         # zona desconocida / fuera de NYC
SURCHARGES = ["extra", "mta_tax", "tolls_amount", "improvement_surcharge",
              "congestion_surcharge", "Airport_fee", "cbd_congestion_fee"]

# Umbrales de calidad heredados del avance 2. Aquí NO eliminan viajes: la
# etiqueta (propina / tarifa) sigue siendo válida aunque la duración o la
# distancia estén mal registradas. Se marcan y el valor dudoso pasa a faltante.
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
MAX_DISTANCE_MI = 100.0
MAX_SPEED_MPH = 80.0

NUMERIC_FEATURES = [
    "fare_amount", "log_fare", "trip_distance", "duration_min", "speed_mph", "passenger_count",
    *SURCHARGES, "subtotal_pre_tip", "ratio_subtotal_fare",
    "hour_sin", "hour_cos", "is_weekend", "is_holiday",
    "pu_airport", "do_airport", "same_zone", "unknown_zone",
    "ratecode_missing", "registro_sospechoso",
]
ONEHOT_FEATURES = ["pickup_hour", "pickup_weekday", "RatecodeID_cat", "VendorID_cat"]
TARGET_ENC_FEATURES = ["PULocationID", "DOLocationID", "route"]
ALL_FEATURES = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENC_FEATURES
FORBIDDEN = {"tip_amount", "total_amount", "tip_percentage", "payment_type"}
assert not FORBIDDEN & set(ALL_FEATURES)


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de registros sospechosos (reglas del avance 2, sin excluir filas)."""
    duration = df["duration_min"]
    distance = df["trip_distance"]
    speed = distance / (duration.clip(lower=1e-3) / 60)
    return pd.DataFrame({
        f"duracion_menor_{MIN_DURATION_MIN:g}_min": duration.lt(MIN_DURATION_MIN),
        f"duracion_mayor_{MAX_DURATION_MIN:g}_min": duration.gt(MAX_DURATION_MIN),
        "distancia_cero_o_faltante": ~distance.gt(0),
        f"distancia_mayor_{MAX_DISTANCE_MI:g}_mi": distance.gt(MAX_DISTANCE_MI),
        f"velocidad_mayor_{MAX_SPEED_MPH:g}_mph": speed.gt(MAX_SPEED_MPH),
    }, index=df.index)


def _holidays(years) -> set:
    cal = USFederalHolidayCalendar()
    return set(cal.holidays(start=f"{min(years)}-01-01", end=f"{max(years)}-12-31").date)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables deterministas fila a fila (no usan el objetivo ni otras filas)."""
    out = pd.DataFrame(index=df.index)
    ts = df["tpep_pickup_datetime"]
    flags = quality_flags(df)
    suspicious = flags.any(axis=1)

    fare = df["fare_amount"].astype("float64")
    distance = df["trip_distance"].where(df["trip_distance"].between(0, MAX_DISTANCE_MI))
    duration = df["duration_min"].where(df["duration_min"].between(MIN_DURATION_MIN, MAX_DURATION_MIN))
    speed = distance / (duration / 60)
    out["fare_amount"] = fare
    out["log_fare"] = np.log1p(fare)
    out["trip_distance"] = distance
    out["duration_min"] = duration
    out["speed_mph"] = speed.where(speed.le(MAX_SPEED_MPH))
    out["passenger_count"] = df["passenger_count"].where(df["passenger_count"].ge(0))
    for column in SURCHARGES:
        out[column] = df[column].astype("float64")
    out["subtotal_pre_tip"] = fare + df[SURCHARGES].fillna(0).sum(axis=1)
    out["ratio_subtotal_fare"] = out["subtotal_pre_tip"] / fare

    minutes = ts.dt.hour * 60 + ts.dt.minute
    out["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    out["is_weekend"] = df["pickup_weekday"].ge(5).astype("int8")
    out["is_holiday"] = ts.dt.date.isin(_holidays(ts.dt.year.unique())).astype("int8")
    out["pu_airport"] = df["PULocationID"].isin(AIRPORT_ZONES).astype("int8")
    out["do_airport"] = df["DOLocationID"].isin(AIRPORT_ZONES).astype("int8")
    out["same_zone"] = df["PULocationID"].eq(df["DOLocationID"]).astype("int8")
    out["unknown_zone"] = (df["PULocationID"].isin(UNKNOWN_ZONES)
                           | df["DOLocationID"].isin(UNKNOWN_ZONES)).astype("int8")
    # Faltante estructural (~27%, concentrado por proveedor): categoría propia + indicador.
    out["ratecode_missing"] = df["RatecodeID"].isna().astype("int8")
    out["registro_sospechoso"] = suspicious.astype("int8")

    out["pickup_hour"] = df["pickup_hour"].astype(str)
    out["pickup_weekday"] = df["pickup_weekday"].astype(str)
    ratecode = df["RatecodeID"].fillna(-1).astype(int)
    out["RatecodeID_cat"] = ratecode.where(ratecode.isin([1, 2, 3, 4, 5, 6, 99, -1]), 99).astype(str)
    out["VendorID_cat"] = df["VendorID"].astype(str)
    out["PULocationID"] = df["PULocationID"].astype(int)
    out["DOLocationID"] = df["DOLocationID"].astype(int)
    out["route"] = out["PULocationID"] * 1000 + out["DOLocationID"]
    return out[ALL_FEATURES]


def _target_encoder(seed: int) -> TargetEncoder:
    return TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=seed)


# Ablación: variables nuevas de la entrega final que describen el cobro previo a la propina.
CHARGE_FEATURES = [*SURCHARGES, "subtotal_pre_tip", "ratio_subtotal_fare"]


def build_preprocessor(scale: bool = False, seed: int = 42, drop=()) -> ColumnTransformer:
    """Preprocesamiento común a todos los modelos. `drop` quita variables numéricas
    (se usa solo en la ablación sin recargos)."""
    scaler = [("escalar", StandardScaler())] if scale else []
    numeric = [c for c in NUMERIC_FEATURES if c not in set(drop)]
    return ColumnTransformer([
        ("numericas", Pipeline([("imputar", SimpleImputer(strategy="median", add_indicator=True))]
                               + scaler), numeric),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                 sparse_output=False), ONEHOT_FEATURES),
        ("target_enc", Pipeline([("te", _target_encoder(seed))] + scaler), TARGET_ENC_FEATURES),
    ], verbose_feature_names_out=False)


class ClippedTargetRegressor(RegressorMixin, BaseEstimator):
    """Recorta el objetivo de ENTRENAMIENTO en su cuantil `upper_q` y no permite
    predicciones negativas.

    La cola del porcentaje de propina llega a 450 000% por tarifas casi nulas.
    Esos valores desestabilizan la regresión cuadrática y el target encoding.
    El cuantil se aprende solo con el entrenamiento; validación y test se
    evalúan SIEMPRE con la etiqueta original, sin recorte.
    """

    def __init__(self, estimator=None, upper_q: float = 0.999):
        self.estimator = estimator
        self.upper_q = upper_q

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.cap_ = float(np.quantile(y, self.upper_q)) if self.upper_q else float("inf")
        self.estimator_ = clone(self.estimator).fit(X, np.clip(y, 0, self.cap_))
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X), 0, None)
