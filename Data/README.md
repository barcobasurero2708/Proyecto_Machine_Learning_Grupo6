# Datos

Se utilizan los cuatro archivos originales entregados por el equipo:

- `yellow_tripdata_2026-01.parquet`
- `yellow_tripdata_2026-02.parquet`
- `yellow_tripdata_2026-03.parquet`
- `yellow_tripdata_2026-04.parquet`

Colócalos en `Data/` en la raíz del proyecto y ejecuta `python main.py --data-dir Data`.
Si `--data-dir` no se indica, `main.py` busca en `Data/` y luego en `data/raw/`.
Se conservaron los nombres de columnas originales, incluida `Airport_fee`.

Los datos pesan aproximadamente 255.6 MB en formato Parquet. No se incluyen otra vez en el paquete de entrega ni se suben a Git. El proyecto original del equipo ya los contiene.

## Descarga desde cero

Fuente verificable: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

```bash
python descargar_datos.py
```

El script descarga los cuatro meses desde los enlaces de distribución de la TLC y comprueba sus huellas SHA-256 contra `manifest.csv`. No sobrescribe archivos existentes. Si la TLC revisa los archivos, la comprobación avisa de la diferencia; no promete reproducir las mismas métricas con una revisión distinta. Conservar la versión original del equipo para reproducir exactamente esta entrega.

También se pueden descargar manualmente en la página oficial, sección 2026, archivos Yellow Taxi de enero a abril.

## Acceso y condiciones de uso

Los archivos son de descarga pública y no requieren una cuenta para esta ruta. La página consultada no declara una licencia estándar específica como CC0 o CC BY para estos Parquet; no se les atribuye una licencia inventada. Se citan la fuente y las condiciones publicadas por NYC:

- [Términos de NYC.gov](https://www.nyc.gov/main/terms-of-use)
- [Términos de NYC Open Data](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje/)
- [Diccionario oficial](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

Fecha de consulta de la documentación: 23 de septiembre de 2026. Fecha original de descarga de los archivos: no proporcionada por el equipo.

## Reserva temporal

- Entrega previa: se leyeron enero-marzo; de abril solo metadatos, esquema y huella.
- Entrega final: enero se usa para ajustar la búsqueda de hiperparámetros, febrero para validarla, enero-febrero para entrenar y
  marzo para elegir el modelo. **Abril se lee únicamente en `python main.py --etapa test`**, que toma el modelo ya fijado en
  `outputs/metrics_validacion.json` y lo evalúa una sola vez. Ninguna decisión de modelado usa resultados de abril.
