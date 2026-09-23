"""Auditoría mensual, selección explícita de viajes y reserva de abril."""
from pathlib import Path
import gc
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SEED = 42
SAMPLE_FRAC = 0.05
MONTHS = ("2026-01", "2026-02", "2026-03", "2026-04")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(data_dir):
    """Solo metadatos y huella de archivos; no se leen registros de abril."""
    rows, schemas = [], []
    for month in MONTHS:
        path = Path(data_dir) / f"yellow_tripdata_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}. Revisa data/README.md")
        parquet = pq.ParquetFile(path)
        schemas.append(parquet.schema_arrow)
        rows.append({"mes": month, "archivo": path.name,
                     "filas": parquet.metadata.num_rows,
                     "columnas": parquet.metadata.num_columns,
                     "bytes": path.stat().st_size, "sha256": sha256(path),
                     "uso": "reserva_final" if month == "2026-04" else
                            ("validacion" if month == "2026-03" else "entrenamiento")})
    if not all(s.equals(schemas[0], check_metadata=False) for s in schemas):
        raise ValueError("Los esquemas difieren. Revisarlos antes de concatenar.")
    return pd.DataFrame(rows)


def explore(data_dir, sample_frac=SAMPLE_FRAC, seed=SEED):
    """Lee enero-marzo, un mes a la vez. Baselines conservan todos los y.

    Muestreo aleatorio simple del 5% en cada mes, después de los filtros.
    Se usa solo para gráficos con variables y para ajustar la regresión lineal.
    La auditoría y la evaluación en marzo usan todas las filas correspondientes.
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac debe estar en (0, 1].")
    inv = inventory(data_dir)
    schemas, missing, payment, flags, flows, stats = [], [], [], [], [], []
    samples, targets, extremes, month_summary = [], {}, [], []
    validation = None
    for month in MONTHS[:3]:
        print(f"Procesando {month}...", flush=True)
        df = pd.read_parquet(Path(data_dir) / f"yellow_tripdata_{month}.parquet")
        n = len(df)
        for column in df.columns:
            schemas.append({"mes": month, "variable": column, "tipo_pandas": str(df[column].dtype)})
            count = int(df[column].isna().sum())
            missing.append({"mes": month, "variable": column,
                            "faltantes": count, "filas": n, "porcentaje": 100 * count / n})
        grouped = df.assign(propina_cero=df.tip_amount.eq(0)).groupby("payment_type", dropna=False)
        for code, group in grouped:
            payment.append({"mes": month, "payment_type": int(code), "n": len(group),
                            "sin_propina_registrada": int(group.propina_cero.sum()),
                            "porcentaje_cero": 100 * group.propina_cero.mean()})
        del grouped, group

        pickup = df.tpep_pickup_datetime
        dropoff = df.tpep_dropoff_datetime
        duration = (dropoff - pickup).dt.total_seconds() / 60
        start = pd.Timestamp(month)
        end = start + pd.offsets.MonthBegin(1)
        in_month = pickup.ge(start) & pickup.lt(end)
        # Se exige disponer de la etiqueta antes de la siguiente partición.
        cutoff = pd.Timestamp("2026-03-01" if month != "2026-03" else "2026-04-01")
        conditions = {
            "fecha_fuera_del_mes_o_nula": ~in_month,
            "tarifa_no_positiva_o_no_finita": ~np.isfinite(df.fare_amount) | df.fare_amount.le(0),
            "propina_negativa_o_no_finita": ~np.isfinite(df.tip_amount) | df.tip_amount.lt(0),
            "duracion_no_positiva_o_no_finita": ~np.isfinite(duration) | duration.le(0),
            "duracion_mayor_180_min": duration.gt(180),
            "distancia_negativa_o_no_finita": ~np.isfinite(df.trip_distance) | df.trip_distance.lt(0),
            "distancia_mayor_200_millas": df.trip_distance.gt(200),
            "tarifa_positiva_menor_1_dolar": df.fare_amount.gt(0) & df.fare_amount.lt(1),
        }
        for name, mask in conditions.items():
            flags.append({"mes": month, "alerta": name, "n": int(mask.sum()),
                          "base": n, "porcentaje": 100 * mask.mean()})

        keep = pd.Series(True, index=df.index)
        flows.append({"mes": month, "paso": "01_originales", "restantes": n, "excluidos_en_paso": 0})
        rules = [
            ("02_fecha_del_mes", in_month),
            ("03_pago_tarjeta", df.payment_type.eq(1)),
            ("04_tarifa_positiva_finita", np.isfinite(df.fare_amount) & df.fare_amount.gt(0)),
            ("05_propina_no_negativa_finita", np.isfinite(df.tip_amount) & df.tip_amount.ge(0)),
            ("06_duracion_positiva_finita", np.isfinite(duration) & duration.gt(0)),
            ("07_etiqueta_antes_del_corte", dropoff.lt(cutoff)),
        ]
        for label, rule in rules:
            previous = int(keep.sum())
            keep &= rule
            flows.append({"mes": month, "paso": label, "restantes": int(keep.sum()),
                          "excluidos_en_paso": previous - int(keep.sum())})
        # Comparación exacta sobre las 20 columnas, antes de variables derivadas.
        clean = df.loc[keep].copy()
        duplicate = clean.duplicated(keep="first")
        n_duplicate = int(duplicate.sum())
        clean = clean.loc[~duplicate].copy()
        flows.append({"mes": month, "paso": "08_sin_duplicados_exactos",
                      "restantes": len(clean), "excluidos_en_paso": n_duplicate})
        clean["duration_min"] = (clean.tpep_dropoff_datetime - clean.tpep_pickup_datetime).dt.total_seconds() / 60
        clean["tip_percentage"] = 100 * clean.tip_amount / clean.fare_amount
        finite_target = np.isfinite(clean.tip_percentage)
        n_invalid_target = int((~finite_target).sum())
        clean = clean.loc[finite_target].copy()
        flows.append({"mes": month, "paso": "09_objetivo_finito", "restantes": len(clean),
                      "excluidos_en_paso": n_invalid_target})
        if clean.empty:
            raise ValueError(f"No quedaron registros elegibles en {month}.")
        clean["mes"] = month
        clean["pickup_hour"] = clean.tpep_pickup_datetime.dt.hour
        clean["pickup_weekday"] = clean.tpep_pickup_datetime.dt.dayofweek
        clean["pickup_date"] = clean.tpep_pickup_datetime.dt.normalize()
        y = clean.tip_percentage.to_numpy(copy=True)
        targets[month] = y
        description = clean.tip_percentage.describe(percentiles=[.01, .25, .5, .75, .95, .99, .999]).to_dict()
        stats.append({"mes": month, **description, "porcentaje_cero": 100 * (y == 0).mean(),
                      "mayores_100": int((y > 100).sum()), "mayores_1000": int((y > 1000).sum())})
        extremes.append(clean.nlargest(5, "tip_percentage")[["mes", "tpep_pickup_datetime", "fare_amount", "tip_amount", "tip_percentage", "duration_min", "trip_distance"]])
        # Fracción idéntica en los tres estratos: conserva aproximadamente su peso.
        sample = clean.sample(frac=sample_frac, random_state=seed + int(month[-2:])).copy()
        samples.append(sample)
        month_summary.append({"mes": month, "originales": n, "elegibles": len(clean),
                              "muestra": len(sample), "duplicados_exactos": n_duplicate,
                              "pickup_min_original": str(pickup.min()), "pickup_max_original": str(pickup.max()),
                              "pickup_min_elegible": str(clean.tpep_pickup_datetime.min()),
                              "pickup_max_elegible": str(clean.tpep_pickup_datetime.max()),
                              "dropoff_max_elegible": str(clean.tpep_dropoff_datetime.max())})
        if month == "2026-03":
            validation = clean[["VendorID", "RatecodeID", "PULocationID", "DOLocationID", "passenger_count",
                                "fare_amount", "trip_distance", "duration_min", "pickup_hour", "pickup_weekday",
                                "tip_percentage"]].copy()
        del df, clean, pickup, dropoff, duration, conditions, rules, keep, sample
        gc.collect()
    return {"inventario": inv, "tipos": pd.DataFrame(schemas), "faltantes": pd.DataFrame(missing),
            "pagos": pd.DataFrame(payment), "alertas": pd.DataFrame(flags), "filtros": pd.DataFrame(flows),
            "objetivo": pd.DataFrame(stats), "resumen_mensual": pd.DataFrame(month_summary),
            "extremos": pd.concat(extremes, ignore_index=True), "muestra": pd.concat(samples, ignore_index=True),
            "targets": targets, "validacion": validation, "sample_frac": sample_frac, "seed": seed}


def save_tables(result, root):
    output = Path(root) / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    for key in ["inventario", "tipos", "faltantes", "pagos", "alertas", "filtros", "objetivo", "resumen_mensual", "extremos"]:
        result[key].to_csv(output / f"{key}.csv", index=False)
