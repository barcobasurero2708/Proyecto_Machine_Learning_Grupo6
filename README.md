# Grupo 6 | Proyecto final de Machine Learning

Predicción del **porcentaje de propina registrada sobre la tarifa base** al cerrar viajes de taxi amarillo pagados con tarjeta en
Nueva York (NYC TLC, enero-abril de 2026).

- **Informe final:** [`reports/informe_final.md`](reports/informe_final.md)
- **Presentación:** [`reports/presentacion_final.pdf`](reports/presentacion_final.pdf) (fuente editable: `reports/presentacion_final.pptx`)
- **Propuesta (entrega previa):** [`proposal.md`](proposal.md)
- **Tablas generadas por el código:** [`reports/resultados.md`](reports/resultados.md) y [`outputs/metrics.json`](outputs/metrics.json)

## Resultado principal

Partición temporal: enero-febrero entrenamiento, marzo validación, abril test (evaluado una sola vez). Errores en puntos porcentuales (pp).

| Modelo | MAE validación (marzo) | MAE test (abril) | MedAE test |
|---|---|---|---|
| Mediana constante (entrega previa) | 9.685 | 10.054 | 5.919 |
| Regresión lineal (entrega previa) | 7.524 | 7.968 | 4.395 |
| **Gradient boosting, pérdida absoluta (final)** | **6.314** | **6.706** | **0.152** |

El modelo final reduce el MAE de test en 15.8% frente al baseline lineal y en 33.3% frente a la mediana. La variable más importante
es la relación subtotal antes de propina / tarifa base: muchos pasajeros eligen 20, 25 o 30% del subtotal en la pantalla de pago.

## Estructura

```
├── README.md
├── proposal.md                       # propuesta de la entrega previa
├── requirements.txt                  # dependencias con versiones
├── main.py                           # punto de entrada: exploración, validación y test
├── ejecutar_notebook.py              # ejecuta los 4 notebooks y guarda sus salidas
├── descargar_datos.py                # descarga opcional desde la TLC y verifica SHA-256
├── Data/
│   ├── README.md                     # fuente, licencia, acceso y reserva temporal
│   └── manifest.csv                  # filas, tamaño y huella de cada Parquet
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb  # entrega previa: EDA, población y baselines
│   ├── 02_limpieza_features.ipynb    # calidad, faltantes, variables y leakage
│   ├── 03_modelos.ipynb              # búsqueda, comparación, ablación y test
│   └── 04_analisis_errores.ipynb     # interpretación, errores por segmento y sesgos
├── src/
│   ├── data.py                       # auditoría (entrega previa) y población de modelado
│   ├── features.py                   # variables, preprocesamiento y control de leakage
│   ├── model.py                      # baselines de la entrega previa (sin cambios)
│   ├── train.py                      # búsqueda, entrenamiento, validación y test
│   ├── evaluate.py                   # métricas, errores por segmento y figuras
│   └── plots.py                      # figuras 01-08 de la exploración
├── outputs/                          # métricas (JSON) y tablas (CSV) regenerables
└── reports/
    ├── informe_final.md
    ├── presentacion_final.pdf / .pptx
    ├── resultados.md                 # tablas generadas por main.py
    └── figures/                      # figuras 01-20
```

## Cómo reproducir

Entorno comprobado: Python 3.12 (también corre en 3.13), 64 bits, 8 GB de RAM o más.

**1. Entorno** (PowerShell, en la carpeta del proyecto):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux/macOS: `python3.12 -m venv .venv` y `.venv/bin/python -m pip install -r requirements.txt`.

**2. Datos.** Colocar en `Data/` los cuatro archivos `yellow_tripdata_2026-0{1,2,3,4}.parquet` (no se suben a Git), o descargarlos y
verificar sus huellas:

```powershell
.\.venv\Scripts\python.exe descargar_datos.py --data-dir Data
```

**3. Pipeline completo** (≈25 minutos con 2 núcleos):

```powershell
.\.venv\Scripts\python.exe main.py --data-dir Data --etapa todo
```

| Etapa | Qué hace | Salidas |
|---|---|---|
| `exploracion` | Auditoría ene-mar y baselines de la entrega previa | `outputs/{inventario,tipos,faltantes,pagos,alertas,filtros,objetivo,resumen_mensual,extremos,baseline_metrics,sensibilidad_tarifa}.csv`, `outputs/metrics_entrega1.json`, figuras 01-08 |
| `validacion` | Búsqueda (ene→feb), 5 modelos + 3 baselines en marzo, ablación, importancia, dependencia parcial, errores | `outputs/metrics_validacion.json`, `busqueda_hiperparametros.csv`, `errores_por_*`, `importancia_variables.csv`, figuras 09-17 |
| `test` | Reentrena el modelo elegido con ene-mar y evalúa una vez en abril | `outputs/metrics_test.json`, `errores_por_segmento_test.csv`, figuras 18-19 |

Siempre se actualizan `outputs/metrics.json` (resumen final) y `reports/resultados.md`. Las etapas pueden correrse por separado
(`--etapa validacion`, luego `--etapa test`). `--rapido` usa muestras pequeñas para comprobar que todo corre (sus métricas no valen).

**4. Notebooks** (leen las salidas del paso 3):

```powershell
.\.venv\Scripts\python.exe ejecutar_notebook.py --data-dir Data
```

También pueden abrirse en VS Code/Jupyter con el kernel de `.venv`. La figura 20 la genera el notebook 02.

## Decisiones clave

- **Población y partición idénticas a la entrega previa** (pago con tarjeta, tarifa positiva, etc.); los baselines reproducen las
  mismas cifras. Abril solo se lee en la etapa `test`, con el modelo ya fijado en `metrics_validacion.json`.
- **Sin leakage:** `tip_amount`, `total_amount`, `payment_type` nunca entran en X. El subtotal se construye con tarifa + recargos, sin
  `total_amount`. Imputación, escalado y target encoding se ajustan solo con entrenamiento dentro de `Pipeline`.
- **Registros sospechosos** (duración, distancia o velocidad imposibles): se marcan y se analizan, no se eliminan.
- **Extremos del objetivo:** se conservan en todas las métricas; solo la etiqueta de entrenamiento se recorta en su cuantil 0.999.
- **Semilla 42** en muestreo, búsqueda, modelos y target encoding.
