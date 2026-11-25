# Informe final | Predicción del porcentaje de propina en taxis amarillos de Nueva York

**Curso:** Machine Learning · **Grupo 6**
**Integrantes:** Aaron Adriano Romano Castro · Enzo Mathias Calderón Soto · Sebastian Ernesto Chahuara Galdos · Carlos Armando Villanueva Lara
**Repositorio:** `Proyecto_Machine_Learning_Grupo6` · **Datos:** NYC TLC Yellow Taxi, enero-abril de 2026

Todas las cifras de este informe provienen de archivos generados por `python main.py --data-dir Data` (carpeta `outputs/` y tabla
resumen `reports/resultados.md`). Los errores se expresan en **puntos porcentuales (pp)** de la tarifa base.

---

## 1. Resumen ejecutivo

Predecimos, al cerrar el taxímetro de un viaje pagado con tarjeta, qué porcentaje de la tarifa base quedará registrado como
propina. Usamos los 14.9 millones de viajes de enero-abril de 2026 con una partición temporal: enero-febrero para entrenar, marzo
para elegir el modelo y abril reservado para una sola evaluación final.

- El modelo final es un **gradient boosting con pérdida absoluta** (`HistGradientBoostingRegressor`). En la evaluación única de
  abril (2 591 610 viajes) obtiene **MAE = 6.71 pp**, frente a 7.97 pp de la regresión lineal de la entrega previa (**−15.8%**) y
  10.05 pp de la mediana constante (**−33.3%**). En marzo la mejora había sido de 16.1% y 34.8%: la elección fue estable.
- La mediana del error absoluto es de solo **0.15 pp** en abril: para la mitad de los viajes el modelo acierta casi exactamente el
  porcentaje, porque muchos pasajeros eligen un botón fijo de la pantalla de pago (20, 25 o 30% del subtotal).
- La variable más importante es nueva respecto a la propuesta: **subtotal antes de propina / tarifa base**. En enero, el 39% de las
  propinas son exactamente el 20% del subtotal (tarifa + recargos). Los recargos se conocen antes de la propina y se suman sin usar
  `total_amount`, así que no hay leakage.
- El error se concentra en los **viajes sin propina registrada (10%)**, que el modelo no anticipa, y en tarifas casi nulas, donde el
  cociente explota. La RMSE global (107 pp en marzo, 655 pp en abril) refleja esos pocos casos extremos; se informa también la RMSE
  con tarifa ≥ USD 1 (≈24 pp).
- La función de pérdida importó más que la familia: los modelos que estiman la media (ridge, random forest, boosting cuadrático)
  apenas mejoran el MAE del baseline lineal.

## 2. Descripción del problema

**Pregunta:** al finalizar un viaje de taxi amarillo, y suponiendo que el pasajero ya eligió pagar con tarjeta, ¿qué porcentaje de
la tarifa base quedará registrado como propina?

- **Objetivo:** `tip_percentage = 100 * tip_amount / fare_amount` (regresión, continuo y no negativo).
- **Unidad de predicción:** un viaje de taxi amarillo pagado con tarjeta, con tarifa positiva y duración positiva.
- **Instante de predicción:** cierre del taxímetro, antes de introducir la propina. Por eso duración, destino y recargos ya son
  conocidos. Que la elección de tarjeta sea conocida en ese momento es un supuesto operativo (los registros no tienen eventos de la
  terminal).
- **Motivación:** cuantificar cuánto del porcentaje de propina es predecible a partir de características observables del viaje y del
  cobro, como referencia para estudiar patrones agregados (por ejemplo, el efecto de los recargos sobre la propina expresada como
  porcentaje de la tarifa). **No** se plantea usarlo para asignar o negar servicio a pasajeros o zonas.
- **Métricas:** MAE (principal), RMSE (secundaria) y R² (apoyo), las tres definidas en la propuesta. Se agregan MedAE, percentil 90
  del error absoluto y la RMSE con tarifa ≥ USD 1 (diagnóstico de la entrega previa). No se usa MAPE porque hay objetivos iguales a cero.

## 3. Descripción del dataset

| Mes | Filas originales | Viajes elegibles | Uso |
|---|---|---|---|
| 2026-01 | 3 724 889 | 2 210 259 | entrenamiento (y ajuste de la búsqueda) |
| 2026-02 | 3 399 866 | 2 014 817 | entrenamiento (y validación de la búsqueda) |
| 2026-03 | 3 952 451 | 2 565 402 | validación |
| 2026-04 | 3 831 240 | 2 591 610 | test, evaluado una sola vez |

- **Fuente:** NYC Taxi & Limousine Commission, [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
  y su [diccionario oficial](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).
- **Estructura:** 4 Parquet (255.6 MB), 20 columnas: fechas de inicio/fin, zonas (`PULocationID`, `DOLocationID`), proveedor,
  código de tarifa, pasajeros, distancia, tipo de pago, tarifa, recargos, propina y total.
- **Licencia y acceso:** descarga pública sin cuenta. La página no declara una licencia estándar (CC0/CC BY); se citan la fuente y los
  [términos de NYC.gov](https://www.nyc.gov/main/terms-of-use). Huellas SHA-256 en `Data/manifest.csv`.
- **Complejidad:** alto volumen, variables temporales, faltantes estructurales, anomalías (fechas de 2008, duraciones negativas,
  tarifas de USD 0.01), objetivo con cola extrema (hasta 450 000%) y riesgo de leakage por los montos del cobro.
- **Muestreo:** las métricas de validación y test usan **todos** los viajes elegibles del mes. Para entrenar los modelos nuevos se usa
  una muestra aleatoria simple de 400 000 viajes por mes (semilla fija), por memoria y tiempo; la búsqueda de hiperparámetros usa
  250 000 por mes. Los baselines constantes usan todas las etiquetas de entrenamiento y la regresión lineal la misma muestra de 5% de
  la entrega previa (211 254 viajes).
- **Limitaciones:** solo tarjeta (las propinas en efectivo no se registran), cuatro meses de una sola ciudad, sin identificadores de
  pasajero o conductor, fechas sin zona horaria explícita.

## 4. Limpieza y preparación de datos

**Población (sin cambios desde la entrega previa).** Fecha de inicio dentro del mes, `payment_type = 1`, tarifa positiva finita,
propina no negativa finita, duración positiva, etiqueta disponible antes del corte de la partición siguiente, sin duplicados exactos y
objetivo finito. `src/data.py::load_population` reproduce exactamente los conteos de `outputs/filtros.csv` (el notebook 02 lo
comprueba con un `assert`).

**Registros sospechosos: marcar, no eliminar.** El segundo avance propuso eliminar viajes con duración < 1 min o > 180 min,
distancia 0 o > 100 mi y velocidad > 80 mph. Esas reglas detectan errores de duración y distancia, no de la etiqueta, y eliminarlos
habría cambiado la población y la comparabilidad con la entrega previa. En la versión final esos valores pasan a faltante, se agrega
el indicador `registro_sospechoso` y el grupo se analiza aparte. Afectan a ~1% de cada mes (`outputs/limpieza_reporte.csv`).

**Faltantes.** Los faltantes de `RatecodeID`, `passenger_count`, `store_and_fwd_flag`, `congestion_surcharge` y `Airport_fee`
(27.6% de los registros originales) pertenecen todos a `payment_type = 0` (Flex Fare), excluido de la población. Dentro de la
población con tarjeta no hay faltantes en esas columnas. El pipeline mantiene un tratamiento seguro: imputación por mediana de
entrenamiento con indicador y categoría propia para `RatecodeID`.

**Extremos del objetivo.** Se conservan en todas las evaluaciones. Solo al **ajustar** los modelos nuevos se recorta la etiqueta de
entrenamiento en su cuantil 0.999 (aprendido con entrenamiento), porque los valores de miles de por ciento desestabilizan la pérdida
cuadrática y el target encoding (`ClippedTargetRegressor`).

**Variables** (`src/features.py`, lista permitida `ALL_FEATURES`):

| Grupo | Variables |
|---|---|
| Tarifa y viaje | `fare_amount`, `log_fare`, `trip_distance`, `duration_min`, `speed_mph`, `passenger_count` |
| Cobro antes de propina (nuevo) | 7 recargos, `subtotal_pre_tip`, `ratio_subtotal_fare` |
| Tiempo | hora cíclica, hora y día (one-hot), fin de semana, feriado federal |
| Zonas | aeropuerto de origen/destino, misma zona, zona desconocida; `PULocationID`, `DOLocationID` y ruta con target encoding |
| Calidad | `ratecode_missing`, `registro_sospechoso`, `RatecodeID` y `VendorID` (one-hot) |

**Control de leakage.** `tip_amount`, `total_amount`, `tip_percentage` y `payment_type` nunca entran en X (comprobado con `assert`).
El subtotal se construye como tarifa + recargos, sin `total_amount`. Imputación, escalado, categorías frecuentes y target encoding
viven dentro de un `Pipeline` ajustado solo con entrenamiento; el `TargetEncoder` usa cross-fitting de 5 particiones sobre el propio
entrenamiento. Los recargos se habían dejado fuera en la propuesta "hasta confirmar que se determinan antes de la propina": el
diccionario los describe como cargos de la tarifa, y los picos exactos en 20/25/30% del subtotal (figura 20) muestran que la
pantalla de propina se calcula sobre ese monto ya conocido.

## 5. Análisis exploratorio

Las figuras 01-08 (entrega previa) documentan volumen, faltantes, distribución del objetivo, pago, variación mensual, hora y tarifa.
Hallazgos adicionales de la fase final:

- **El objetivo es estable entre meses** (figura 09, `outputs/deriva_mensual.csv`): mediana 26.6, 26.3 y 26.4% en enero, febrero y
  marzo; la proporción de propinas cero sube levemente (9.5 → 9.6 → 9.9%) y en abril llega a 10.1%.
- **Propina sobre subtotal** (figura 20): mediana exactamente 20%; 39.0% de los viajes en 20.00%, 5.7% en 25% y 2.4% en 30% (±0.05).
- **`RatecodeID = 99`** (código desconocido, ~5% de enero) tiene mediana de propina 0.
- **Proveedores:** con `VendorID = 1` el 25% de los viajes registra propina cero, frente a 5% con `VendorID = 2`. El pasajero no
  elige el proveedor, así que la diferencia apunta a cómo cada sistema registra la propina (sesgo de medición, sección 13).

## 6. Estrategia de partición y validación

| Etapa | Ajuste | Evaluación | Decisión que se toma |
|---|---|---|---|
| Búsqueda de hiperparámetros | enero (250 000) | febrero (250 000) | mejor configuración de cada familia |
| Comparación de familias | enero + febrero (800 000) | **marzo completo** (2 565 402) | modelo final |
| Evaluación final | enero-marzo (1 200 000) | **abril completo** (2 591 610), una vez | ninguna |

La partición es temporal en todos los niveles (no hay split aleatorio global). Abril no se lee hasta la etapa `test`, que toma el
modelo y sus hiperparámetros de `outputs/metrics_validacion.json`. Ninguna decisión de modelado se tomó mirando marzo para la
búsqueda ni abril para nada. Semilla 42 en muestreo, modelos, búsqueda y target encoding.

## 7. Baselines

Se mantienen los tres baselines de la entrega previa, sin cambios de código (`src/model.py`), evaluados sobre las mismas filas que
los modelos nuevos. La validación reproduce los números entregados:

| Baseline | Entrenamiento | MAE marzo (pp) | RMSE marzo (pp) | RMSE tarifa ≥ USD 1 |
|---|---|---|---|---|
| Media (25.36%) | 4 225 076 etiquetas | 9.748 | 107.61 | 25.33 |
| Mediana (26.38%) | 4 225 076 etiquetas | 9.685 | 107.62 | 25.36 |
| Regresión lineal (`log1p` + one-hot) | 211 254 viajes | 7.524 | 107.28 | 24.01 |

## 8. Modelos entrenados

Cinco modelos de tres familias, con el mismo preprocesamiento:

| Modelo | Familia | Pérdida |
|---|---|---|
| `ridge` | lineal regularizado | cuadrática |
| `lineal_absoluto` (`SGDRegressor`, ε = 0) | lineal regularizado | absoluta (regresión L1) |
| `random_forest` (60 árboles) | bagging de árboles | cuadrática |
| `hgb_absoluto` | gradient boosting (histogramas) | absoluta |
| `hgb_cuadratico` | gradient boosting (histogramas) | cuadrática |

Se incluyeron las dos pérdidas porque el MAE se minimiza con la **mediana** condicional, y en este objetivo (10% de ceros, cola
extrema) la media y la mediana condicionales se separan mucho.

## 9. Búsqueda de hiperparámetros

Ajuste en enero y validación en febrero (`outputs/busqueda_hiperparametros.csv`, figura 10). Rejilla completa para los modelos
lineales y muestreo aleatorio (semilla 42) para árboles: 5 configuraciones de ridge (α), 4 de L1 (α), 4 de random forest (profundidad,
hojas mínimas, fracción de variables), 6 de boosting absoluto y 4 de boosting cuadrático (tasa de aprendizaje, hojas, hojas mínimas,
L2), con early stopping interno.

| Modelo | Mejor configuración | MAE febrero (pp) |
|---|---|---|
| `hgb_absoluto` | learning_rate 0.2, max_leaf_nodes 31, min_samples_leaf 50, l2 0 | 6.415 |
| `lineal_absoluto` | α = 1e-4 | 6.643 |
| `random_forest` | max_depth 12, min_samples_leaf 20, max_features 0.3 | 7.467 |
| `hgb_cuadratico` | learning_rate 0.05, max_leaf_nodes 31, min_samples_leaf 50, l2 0 | 7.472 |
| `ridge` | α = 0.1 | 7.549 |

Dentro de cada familia los hiperparámetros cambian el MAE en centésimas o pocas décimas; entre pérdidas la diferencia es de ~1 pp.

## 10. Evaluación final

**Validación (marzo completo, figura 11):**

| Modelo | MAE | RMSE | R² | MedAE | P90 error | RMSE tarifa ≥ 1 |
|---|---|---|---|---|---|---|
| baseline_mediana | 9.685 | 107.62 | −0.0001 | 5.990 | 26.38 | 25.36 |
| baseline_lineal | 7.524 | 107.28 | 0.0060 | 4.327 | 17.35 | 24.01 |
| ridge | 7.482 | 107.14 | 0.0087 | 3.992 | 17.08 | 23.93 |
| lineal_absoluto | 6.532 | 105.78 | 0.0336 | 0.977 | 18.95 | 24.08 |
| random_forest | 7.423 | 107.18 | 0.0080 | 3.920 | 16.93 | 23.63 |
| **hgb_absoluto** | **6.314** | 107.23 | 0.0070 | **0.103** | 19.19 | 23.96 |
| hgb_cuadratico | 7.413 | 107.15 | 0.0085 | 3.925 | 16.91 | **23.52** |

Se elige `hgb_absoluto` por menor MAE (−16.1% frente al baseline lineal, −34.8% frente a la mediana).

**Ablación de las variables de cobro:** el mismo modelo sin los recargos, el subtotal ni el ratio obtiene MAE = 6.474 pp. Los recargos
aportan 0.16 pp adicionales; el resto de la mejora (≈1.05 pp frente al baseline lineal) se debe a la pérdida absoluta y al modelo no lineal.

**Test (abril completo, evaluación única, figura 18):**

| Modelo | MAE | RMSE | MedAE | RMSE tarifa ≥ 1 |
|---|---|---|---|---|
| baseline_media | 10.109 | 654.99 | 6.378 | 24.92 |
| baseline_mediana | 10.054 | 654.99 | 5.919 | 24.97 |
| baseline_lineal | 7.968 | 654.92 | 4.395 | 23.63 |
| **hgb_absoluto** | **6.706** | 654.92 | **0.152** | 23.68 |

Mejora en test: −15.8% frente al baseline lineal y −33.3% frente a la mediana, casi idéntica a validación. Todos los modelos empeoran
unas décimas en abril, en línea con el aumento de propinas cero (10.1%). La RMSE de abril crece por unos pocos viajes con tarifa casi
nula y porcentajes enormes; con tarifa ≥ USD 1 queda en el mismo rango que marzo.

**Lectura honesta:** el R² es casi cero en todos los modelos porque la varianza del objetivo está dominada por la cola extrema, y el
modelo final no mejora la RMSE frente a los modelos cuadráticos: optimiza la métrica principal declarada (MAE) a costa de la
secundaria. Es una mejora real pero acotada: el porcentaje de propina depende en gran parte de una decisión del pasajero que no está
en los datos.

## 11. Interpretación

- **Importancia por permutación** (figura 13, 30 000 viajes de marzo): `ratio_subtotal_fare` (+3.17 pp de MAE al permutarla),
  `RatecodeID` (+1.49), `fare_amount` (+0.50), `VendorID` (+0.45) y `subtotal_pre_tip` (+0.19). Hora, duración, distancia y zonas
  aportan muy poco una vez conocidos la tarifa y los recargos.
- **Dependencia parcial** (figura 14): la predicción media sube de ~16% a ~38% cuando el subtotal pasa de 1.0 a 2.5 veces la tarifa
  (el mismo botón de 20% equivale a más porcentaje de la tarifa); baja de ~29.5% a ~26.5% cuando la tarifa pasa de USD 3 a USD 25, y
  es prácticamente plana en la duración.
- **Mecanismo:** el modelo aprende sobre todo cómo la pantalla de pago traduce un porcentaje del subtotal a un porcentaje de la tarifa,
  más los códigos que casi nunca registran propina (`RatecodeID = 99`) y las diferencias entre proveedores.

## 12. Análisis de errores

Tablas completas en `outputs/errores_por_segmento_validacion.csv` y `..._test.csv`; figuras 15-17 y 19.

| Segmento (validación) | % viajes | MAE final | MAE baseline lineal | Sesgo medio |
|---|---|---|---|---|
| Propina real cero | 9.9 | 13.12 | 12.51 | +13.12 |
| Propina real positiva | 90.1 | 5.57 | 6.98 | +1.37 |
| Tarifa USD 1-5 | 1.3 | 30.45 | 33.67 | −14.20 |
| Tarifa USD 20-50 | 21.9 | 4.43 | 6.00 | +2.32 |
| Distancia < 1 mi | 25.1 | 9.01 | 10.38 | +1.60 |
| Madrugada (0-5 h) | 7.0 | 8.77 | 9.96 | +3.53 |
| Desde aeropuerto | 7.5 | 5.47 | 6.97 | +2.61 |
| VendorID 1 | 22.6 | 5.31 | 6.45 | +0.87 |
| VendorID 2 | 77.4 | 6.61 | 7.84 | +3.02 |
| Registro sospechoso | 1.0 | 39.81 | 42.32 | −24.64 |

1. **Propinas cero:** son el error dominante. El modelo predice el porcentaje típico y no anticipa que el pasajero no deje propina; en
   ese segmento es incluso algo peor que el baseline lineal (−4.9% en marzo, −3.2% en abril). Explican casi todo el sesgo positivo.
2. **Tarifas pequeñas:** el denominador amplifica cualquier propina. Los 18 viajes de marzo con tarifa de hasta USD 1 tienen un MAE de
   miles de puntos y determinan la RMSE global. Los 25 casos con mayor error (`outputs/casos_mayor_error_validacion.csv`) son tarifas
   de USD 0.01-3 con propinas de varios dólares, varios con distancia 0 o duración de segundos.
3. **Viajes cortos, madrugada y fin de semana** tienen más error; aeropuerto y viajes largos, menos y con la mayor mejora relativa (≈18-25%).
4. **Zonas:** el MAE por zona de inicio (52 zonas con ≥ 5 000 viajes) va de 5.1 a 9.4 pp y no se explica por su proporción de ceros.
5. **Registros sospechosos:** su MAE es casi siete veces el del resto; conservarlos permite medirlo en vez de ocultarlo.
6. **Test:** los mismos patrones se repiten en abril (`04_analisis_errores.ipynb`, sección 7).

## 13. Riesgos éticos, sesgos y limitaciones

- **Sesgo de observación:** solo se observan propinas con tarjeta; un viaje con propina cero registrada pudo recibir efectivo. El
  objetivo es "propina registrada", no "propina real".
- **Sesgo de medición por proveedor:** la tasa de ceros cambia mucho entre `VendorID` 1 y 2 y el error también. Usar el modelo para
  comparar conductores o flotas confundiría la tecnología del taxímetro con el comportamiento del pasajero.
- **Zonas y discriminación:** predecir menos propina en ciertas zonas podría incentivar a evitar recoger pasajeros allí. El modelo es
  descriptivo; no debe usarse para asignar o negar servicio. Las asociaciones no son causales.
- **Supuesto de disponibilidad:** se asume que la elección de tarjeta y los recargos se conocen antes de la propina. Es coherente con
  el diccionario y con los picos en 20/25/30%, pero no se puede verificar evento a evento.
- **Representatividad:** cuatro meses de invierno-primavera de una ciudad; no cubre toda la estacionalidad.
- **Muestreo para entrenar:** los modelos nuevos usan 400 000 viajes por mes; con más datos podrían mejorar ligeramente.
- **Dependencia entre viajes:** sin identificadores de conductor o pasajero no se pueden separar usuarios recurrentes.
- **Calidad de la tarifa:** en tarifas casi nulas el cociente no representa una propina "porcentual" razonable; el MAE las incluye.

## 14. Trabajo futuro

1. **Modelo en dos partes:** un clasificador de "propina cero" (con métricas para desbalance: PR-AUC, F1) y un regresor para propinas
   positivas, combinados para minimizar el MAE.
2. **Objetivo alternativo:** predecir la propina sobre el subtotal (donde 20% es la moda) y transformarla a porcentaje de la tarifa.
3. Validación progresiva con más meses (rolling origin) y más datos de entrenamiento con procesamiento por lotes.
4. Unir el lookup de zonas para analizar errores por distrito y revisar equidad territorial.
5. Calibrar intervalos de predicción (cuantiles 10-90) en vez de una sola predicción.

## 15. Instrucciones de reproducción

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# colocar los 4 Parquet en Data/ (o: python descargar_datos.py --data-dir Data)
.\.venv\Scripts\python.exe main.py --data-dir Data --etapa todo
.\.venv\Scripts\python.exe ejecutar_notebook.py --data-dir Data
```

`main.py --etapa todo` ejecuta en orden: exploración de la entrega previa (figuras 01-08), validación (búsqueda, modelos, figuras
09-17) y test (figuras 18-19), y escribe `outputs/metrics.json` y `reports/resultados.md`. En un equipo de 2 núcleos y 8 GB la
validación tarda ~17 minutos y el test ~5. `--rapido` permite una prueba corta. Detalles en `README.md`.

---

### Historial del proyecto

| Entrega | Qué aportó | Qué se conserva en la versión final |
|---|---|---|
| Entrega previa (avance 1) | Problema, población, partición temporal, EDA, baselines | Todo: mismas reglas, mismas cifras de baselines |
| Avance 2 | Pipeline modular (`data/features/train/evaluate`), reglas de calidad, target encoding, tres familias, análisis por segmentos, test único | La estructura y las ideas, adaptadas al objetivo de propina. Las reglas de calidad pasan de eliminar filas a marcarlas |
| Entrega final | Recargos y ratio del subtotal, pérdida absoluta, búsqueda de hiperparámetros, ablación, interpretación, errores y test en abril | — |

El avance 2 se había desarrollado en un repositorio aparte con otro objetivo (duración del viaje) y con salidas que no correspondían a
este proyecto. En la entrega final se integró su metodología en este repositorio y todas las salidas se regeneraron para el
porcentaje de propina, de modo que notebooks, `outputs/` e informe describen el mismo experimento.

