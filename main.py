"""Punto de entrada único del proyecto (Grupo 6, Machine Learning).

Etapas:
    exploracion  Entrega 1: auditoría de ene-mar, figuras 01-08 y baselines (outputs/*.csv)
    validacion   Entrega final: búsqueda de hiperparámetros, 3 familias de modelos,
                 comparación en marzo, interpretación y análisis de errores
    test         Reentrena el modelo elegido con ene-mar y lo evalúa UNA vez en abril
    todo         Las tres etapas en orden

Ejemplos:
    python main.py --data-dir Data --etapa todo
    python main.py --data-dir Data --etapa validacion --rapido   # prueba corta
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def find_data_dir(value):
    if value:
        return Path(value)
    for candidate in (ROOT / "Data", ROOT / "data" / "raw", ROOT / "data"):
        if (candidate / "yellow_tripdata_2026-01.parquet").exists():
            return candidate
    raise SystemExit("No se encontraron los Parquet. Ver Data/README.md")


def run_exploration(data_dir):
    from src.data import SAMPLE_FRAC, SEED, explore, save_tables
    from src.model import run_baselines
    from src.plots import create_figures
    result = explore(data_dir, sample_frac=SAMPLE_FRAC, seed=SEED)
    save_tables(result, ROOT)
    run_baselines(result, ROOT)
    create_figures(result, ROOT)


def _md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_summary():
    """outputs/metrics.json (resumen final) y reports/resultados.md (tablas generadas)."""
    out = ROOT / "outputs"
    combined, md = {}, ["# Resultados generados por `main.py`", "",
                        "_Archivo generado automáticamente. No editar a mano: el informe final cita estas tablas._", ""]
    e1 = out / "metrics_entrega1.json"
    if e1.exists():
        combined["entrega1_baselines_validacion_marzo"] = json.loads(e1.read_text(encoding="utf-8"))["metricas"]
    search = out / "busqueda_hiperparametros.csv"
    if search.exists():
        combined["busqueda_hiperparametros"] = pd.read_csv(search).to_dict(orient="records")
    vp, tp = out / "metrics_validacion.json", out / "metrics_test.json"
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        combined["validacion"] = v
        t = pd.DataFrame(v["resultados"]).T.drop(columns=["params"], errors="ignore").reset_index(names="modelo")
        md += [f"## Validación: entrenamiento {', '.join(v['split']['entrenamiento'])} → marzo",
               f"Viajes de validación: {v['n_validacion']:,}. Métricas en puntos porcentuales (pp).", "",
               _md_table(t), "", f"**Modelo elegido por MAE:** `{v['mejor_modelo']}` "
               f"{json.dumps(v['parametros_elegidos'][v['mejor_modelo']])}. Mejora de MAE frente al baseline "
               f"lineal: **{v['mejora_mae_vs_baseline_lineal_pct']}%**; frente a la mediana: "
               f"**{v['mejora_mae_vs_baseline_mediana_pct']}%**.", ""]
        md += ["## Búsqueda de hiperparámetros (ajuste en enero, validación en febrero)", "",
               _md_table(pd.read_csv(search)), ""]
    for name, title in [("limpieza_reporte.csv", "Registros sospechosos marcados (no se eliminan)"),
                        ("deriva_mensual.csv", "Objetivo por mes (poblaciones completas)"),
                        ("importancia_variables.csv", "Importancia por permutación"),
                        ("errores_por_segmento_validacion.csv", "Errores por segmento (validación)")]:
        if (out / name).exists():
            md += [f"## {title}", "", _md_table(pd.read_csv(out / name)), ""]
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf-8"))
        combined["test"] = t
        tt = pd.DataFrame(t["resultados"]).T.reset_index(names="modelo")
        md += [f"## Test único: entrenamiento ene-mar → abril ({t['n_test']:,} viajes)", "", _md_table(tt), "",
               f"Mejora de MAE del modelo final frente al baseline lineal en test: "
               f"**{t['mejora_mae_vs_baseline_lineal_pct']}%**; frente a la mediana: "
               f"**{t['mejora_mae_vs_baseline_mediana_pct']}%**.", ""]
        if (out / "errores_por_segmento_test.csv").exists():
            md += ["## Errores por segmento (test)", "",
                   _md_table(pd.read_csv(out / "errores_por_segmento_test.csv")), ""]
    (out / "metrics.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "resultados.md").write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="carpeta con los cuatro Parquet (por defecto Data/)")
    parser.add_argument("--etapa", choices=["exploracion", "validacion", "test", "todo"], default="todo")
    parser.add_argument("--n-por-mes", type=int, default=400_000, help="viajes por mes para entrenar modelos")
    parser.add_argument("--n-busqueda", type=int, default=250_000, help="viajes por mes en la búsqueda")
    parser.add_argument("--rf-filas", type=int, default=150_000, help="filas por árbol del Random Forest")
    parser.add_argument("--n-importancia", type=int, default=30_000)
    parser.add_argument("--rapido", action="store_true", help="muestras pequeñas para comprobar que corre")
    args = parser.parse_args()
    args.data_dir = find_data_dir(args.data_dir)
    if args.rapido:
        args.n_por_mes, args.n_busqueda, args.rf_filas, args.n_importancia = 40_000, 20_000, 15_000, 3_000
    np.random.seed(42)
    for folder in (ROOT / "outputs", ROOT / "reports" / "figures"):
        folder.mkdir(parents=True, exist_ok=True)

    if args.etapa in ("exploracion", "todo"):
        run_exploration(args.data_dir)
    if args.etapa in ("validacion", "test", "todo"):
        from src.train import stage_test, stage_validation
        if args.etapa in ("validacion", "todo"):
            stage_validation(args)
        if args.etapa in ("test", "todo"):
            stage_test(args)
    write_summary()


if __name__ == "__main__":
    main()
