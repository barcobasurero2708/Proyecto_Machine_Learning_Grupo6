# Datos

Se utilizan los cuatro archivos originales entregados por el equipo:

- `yellow_tripdata_2026-01.parquet`
- `yellow_tripdata_2026-02.parquet`
- `yellow_tripdata_2026-03.parquet`
- `yellow_tripdata_2026-04.parquet`

Colócalos en `Data/` en la raíz del proyecto y ejecuta `python main.py --data-dir Data`.
Otra ubicación admitida es `data/raw/`, predeterminada para los scripts.
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

Se leen enero-marzo para desarrollar la entrega. De abril solo se consultan el tamaño, el esquema, el número de filas en metadatos y la huella del archivo. No se calculan distribuciones, faltantes, filtros ni métricas de abril. Si ya se tomaron decisiones mirando sus resultados con el script original, se requiere otro mes posterior para una prueba final intacta.
