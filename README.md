# Grupo 6 | Entrega 1 de Machine Learning

Predicción del porcentaje de propina registrada, sobre la tarifa base, al finalizar viajes de taxi amarillo pagados con tarjeta en Nueva York.


## Contenido

| Archivo o carpeta | Qué contiene |
|---|---|
| `proposal.md` | Los 13 apartados de la propuesta |
| `notebooks/exploracion_inicial.ipynb` | Exploración ejecutada, tablas, ocho figuras, baselines y discusión |
| `src/` | Carga, selección, modelos y gráficos reutilizables |
| `main.py` | Reproduce resultados y figuras por terminal |
| `ejecutar_notebook.py` | Ejecuta las celdas y guarda sus salidas sin servidor |
| `requirements.txt` | Dependencias con versiones |
| `data/README.md`, `data/manifest.csv` | Fuente, descarga, acceso y huellas de los datos |
| `outputs/` | Inventario, tipos, faltantes, filtros, alertas, métricas y resultados numéricos |
| `reports/figures/` | Ocho figuras en PNG |
| `reports/guia_entrega1.md` | Explicación de resultados y correspondencia con la rúbrica |
| `referencia/main_original.py` | Copia del código recibido, solo para comparar; no es el punto de entrada |

## 1. Preparar Python

Entorno comprobado: **Python 3.12.14, 64 bits**. Conviene usar Python 3.12 en un entorno virtual. Reservar memoria para procesar un mes completo; en esta implementación no se concatenan los cuatro archivos originales. La ejecución se verificó en un entorno con 16 GB de RAM; el consumo y tiempo en tu equipo pueden variar.

En PowerShell de Windows, dentro de la carpeta del proyecto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No necesitas activar el entorno para usar estos comandos. En Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Colocar los cuatro Parquet

Crea `Data/` junto a `main.py` y coloca allí los cuatro archivos que ya tienes. No copies el ZIP dentro de esa carpeta: deben estar los archivos `.parquet` extraídos.

- `Data/yellow_tripdata_2026-01.parquet`
- `Data/yellow_tripdata_2026-02.parquet`
- `Data/yellow_tripdata_2026-03.parquet`
- `Data/yellow_tripdata_2026-04.parquet`

Este paquete no vuelve a incluir los datos porque ya están en el proyecto original. Como alternativa, el script `descargar_datos.py` permite obtener los archivos públicos desde cero y verificar sus huellas:

```powershell
.\.venv\Scripts\python.exe descargar_datos.py --data-dir Data
```

El modo de verificación de archivos existentes se comprobó con los cuatro Parquet recibidos. No se volvió a transferir la descarga remota completa; si la TLC revisa una versión, la comprobación de huellas lo notificará.

## 3. Ejecutar el análisis

```powershell
.\.venv\Scripts\python.exe main.py --data-dir Data
```

Este comando regenera `outputs/` y `reports/figures/` usando los mismos datos y semilla 42. No cambia `proposal.md`, el texto de la guía ni las salidas guardadas del notebook. Si cambias datos, población o modelos, también debes actualizar esos textos.

Para regenerar el notebook y sus salidas:

```powershell
.\.venv\Scripts\python.exe ejecutar_notebook.py --data-dir Data
```

En Linux/macOS usa `.venv/bin/python` en lugar de `.\.venv\Scripts\python.exe`.

También puedes abrir `notebooks/01_exploracion_inicial.ipynb` en VS Code, elegir como kernel el Python de `.venv` y ejecutar todas las celdas en orden. El notebook detecta `Data/`; si la ruta es distinta, define `TAXI_DATA_DIR` o modifica su celda de configuración.

La verificación guardada ejecutó las **16 celdas de código secuencialmente con IPython en un proceso**, sin servidor de Jupyter. Las tablas e imágenes son salidas de la ejecución real. No se usaron resultados inventados o escritos manualmente dentro de las salidas.

## 4. Qué reproduce

- Auditoría completa de **11 077 206 registros de enero-marzo**.
- Objetivo y selección de población, con exclusiones trazables.
- Muestra aleatoria de 339 524 viajes (5% por mes) para gráficos multivariados.
- Media y mediana sobre 4 225 076 viajes de enero-febrero.
- Regresión sobre 211 254 viajes muestreados de entrenamiento.
- Evaluación común sobre 2 565 402 viajes de marzo.
- Abril reservado: solamente inventario, esquema y huella; ninguna métrica de prueba.

| Modelo | Filas de ajuste | MAE (pp) | RMSE (pp) | R² |
| --- | --- | --- | --- | --- |
| Media | 4 225 076 | 9.7479 | 107.6096 | -0.000006 |
| Mediana | 4 225 076 | 9.6848 | 107.6168 | -0.000140 |
| Lineal_log1p | 211 254 | 7.5237 | 107.2835 | 0.006047 |

MAE y RMSE están en **puntos porcentuales**. La gran RMSE se explica en la propuesta y en el diagnóstico `outputs/sensibilidad_tarifa.csv`; no se oculta ni se sustituye por una cifra de una población recortada.

## 5. Decisiones que hay que entender

La predicción ocurre al final del viaje; por eso se conocen duración y destino. El objetivo es propina registrada de tarjeta sobre tarifa base. No se usa `total_amount` ni `tip_amount` como predictor. Los porcentajes altos se conservan en las métricas principales. Las constantes y todas las transformaciones se ajustan solo con entrenamiento.


