# Propuesta | Predicción del porcentaje de propina en taxis amarillos de Nueva York


## 1. Título del proyecto

Predicción del porcentaje de propina registrada en viajes de taxi amarillo pagados con tarjeta en Nueva York.

## 2. Integrantes

- Aaron Adriano Romano Castro
- Enzo Mathias Calderón Soto
- Sebastian Chahuara Galdos
- Carlos Armando Villanueva
## 3. Dataset elegido

NYC Taxi & Limousine Commission (TLC), Yellow Taxi Trip Records. Se utilizan enero, febrero, marzo y abril de 2026. Son **14 908 446 registros y 20 columnas por archivo**, con fechas, zonas, pago, distancia y montos. Los cuatro Parquet ocupan 255.56 MB.

| Mes | Filas originales | Columnas | Uso |
| --- | --- | --- | --- |
| 2026-01 | 3 724 889 | 20 | entrenamiento |
| 2026-02 | 3 399 866 | 20 | entrenamiento |
| 2026-03 | 3 952 451 | 20 | validacion |
| 2026-04 | 3 831 240 | 20 | reserva_final |

La fuente oficial es [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Se consultó el [diccionario oficial](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf). Las huellas de los archivos utilizados están en `data/manifest.csv`.

Acceso: descarga pública, sin cuenta en los enlaces utilizados. La página consultada no especifica una licencia estándar como CC0 o CC BY para estos Parquet; se referencia la fuente y los [términos de NYC.gov](https://www.nyc.gov/main/terms-of-use), sin atribuir permisos no documentados. Los términos de [NYC Open Data](https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje/) se incluyen como referencia complementaria. Fecha de consulta de documentación: 23 de septiembre de 2026; fecha original de descarga del equipo: no informada.

Viabilidad: los cuatro meses cumplen el rango de 3 a 12 meses recomendado en la consigna. El volumen, las variables temporales, los faltantes, las anomalías y el riesgo de fuga de información hacen que el problema requiera preparación y evaluación cuidadosas. Se procesa un mes a la vez y se muestrea únicamente para algunas tareas.

## 4. Pregunta predictiva

**Al finalizar un viaje de taxi amarillo, y suponiendo que el pasajero ya eligió pagar con tarjeta, ¿qué porcentaje de la tarifa base quedará registrado como propina?**

Se busca establecer si las características observadas del viaje permiten superar una predicción constante. Esto aporta una referencia para estudiar patrones agregados de propina registrada; no se plantea asignar servicio ni rechazar pasajeros según la predicción.

El instante propuesto es el cierre del taxímetro, antes de introducir la propina. La disponibilidad de la elección de tarjeta en ese momento es un supuesto operativo: los registros históricos no contienen eventos que permitan comprobarlo. Sin acceso a ese dato previo, el análisis debe describirse como estimación retrospectiva condicionada al pago con tarjeta.

## 5. Variable objetivo

`tip_percentage = 100 * tip_amount / fare_amount`.

Ejemplo: USD 4 de propina sobre USD 20 de tarifa base equivalen a 20%. El denominador es `fare_amount`, no `total_amount`: la proporción no representa la propina sobre el cobro total con impuestos, recargos y peajes.

Es una variable continua no negativa. Se conservan ceros y valores superiores al 100%; un valor grande no se considera automáticamente imposible. Tarifas nulas, no positivas o no finitas impiden construir un cociente útil para esta población. Propinas negativas o no finitas se excluyen y se documentan como registros no compatibles con una propina ordinaria no negativa.

## 6. Unidad de predicción

Un viaje individual de taxi amarillo, del mes indicado por su fecha real de inicio, pagado con tarjeta (`payment_type == 1`), con tarifa positiva, objetivo finito y duración positiva. Los códigos de otros medios de pago no se mezclan con tarjeta. La TLC señala que `tip_amount` no incluye propinas en efectivo; por ello las conclusiones no se generalizan a esos viajes.

Se eliminan duplicados exactos sobre las 20 columnas originales. No se encontraron duplicados en las poblaciones elegibles de enero-marzo. Al no haber un identificador único de viaje, no se asegura que todo registro similar sea un duplicado.

| Mes | Originales | Elegibles | Muestra (5%) |
| --- | --- | --- | --- |
| 2026-01 | 3 724 889 | 2 210 259 | 110 513 |
| 2026-02 | 3 399 866 | 2 014 817 | 100 741 |
| 2026-03 | 3 952 451 | 2 565 402 | 128 270 |

## 7. Variables disponibles antes de la predicción

Las siguientes variables solo se justifican bajo el escenario **al finalizar el viaje**:

| Variable | Uso inicial | Justificación |
|---|---|---|
| `fare_amount` | Numérica | Tarifa base ya calculada al cerrar el taxímetro |
| `trip_distance` | Numérica | Distancia recorrida, en millas |
| `duration_min` | Numérica derivada | Minutos entre inicio y fin del viaje |
| `passenger_count` | Numérica | Recuento registrado de pasajeros |
| `VendorID` | Categórica | Proveedor del registro, no identificador del conductor |
| `RatecodeID` | Categórica | Código de tarifa al finalizar |
| `PULocationID`, `DOLocationID` | Categóricas | Zonas de inicio y fin |
| `pickup_hour` | Categórica derivada | Hora del inicio registrado |
| `pickup_weekday` | Categórica derivada | Día de semana del inicio registrado |

Las fechas se usan para derivar variables y separar periodos, no como enteros sin interpretación. Los códigos de zona no representan distancias numéricas. Los baselines constantes usan únicamente la distribución del objetivo de entrenamiento.

`extra`, `mta_tax`, `tolls_amount`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee` y `cbd_congestion_fee` quedan fuera del primer modelo para mantenerlo simple; se puede estudiar su inclusión posterior si se confirma que están determinados antes de registrar la propina. `store_and_fwd_flag` se excluye porque describe el envío del registro y su disponibilidad anticipada no está establecida.

## 8. Riesgos de leakage

| Riesgo | Ejemplo | Control |
|---|---|---|
| Respuesta directa | Usar `tip_amount` para predecir su propio porcentaje | Solo se usa para crear la etiqueta |
| Cobro dependiente de la respuesta | Incluir `total_amount` o reconstrucciones que revelan propina | Se excluyen de la lista de predictores |
| Momento de predicción incorrecto | Usar duración o destino si se promete predecir al inicio | Se fija el instante al final; un caso al inicio requiere rediseño |
| Transformaciones con información futura | Imputar o aprender categorías con todos los meses | `Pipeline` ajustado solo en enero-febrero |
| Etiquetas aún no disponibles | Entrenar con viajes que terminan después del corte | Se exige finalización antes del periodo siguiente |
| Prueba final usada para decidir | Elegir un modelo tras mirar abril repetidamente | Marzo sirve para desarrollo y abril se reserva |
| Filtro temporal incompleto | Usar `.dt.month <= 3` e incorporar otro año | Se comprueban intervalos completos con año y mes |



## 9. Métrica principal y secundaria

- **Principal: MAE**, promedio del error absoluto, expresado en puntos porcentuales (pp). Permite explicar el error de una predicción de porcentaje sin convertirlo en porcentaje relativo.
- **Secundaria: RMSE**, también en pp. Da más peso a errores grandes y permite observar el efecto de la cola extrema.
- **Complementaria: R²**. No sustituye las dos métricas principales. Puede ser negativo para un baseline constante porque su valor se aprendió con otro periodo.

No se usa MAPE: existen objetivos cero. Tampoco se interpreta un MAE de 7.5 pp como “92.5% de accuracy”; esta tarea no es clasificación.

## 10. Plan de validación

Enero-febrero: entrenamiento. Marzo: validación. Abril: prueba final reservada. Los viajes de entrenamiento deben terminar antes del 1 de marzo y los de validación antes del 1 de abril. Solo se inspeccionan metadatos del archivo reservado.

La partición temporal simula el uso de datos pasados para estimar viajes futuros. No se realiza un split aleatorio global. En siguientes semanas se usarán cortes internos cronológicos o validación progresiva dentro de enero-febrero para ajustar hiperparámetros y se limitarán las comparaciones sobre marzo. Al cerrar el diseño, se podrá reentrenar con enero-marzo y evaluar una vez en abril.

Limitación: no hay identificadores de conductor o pasajero, por lo que no se pueden aislar personas repetidas. Las fechas carecen de zona horaria explícita; se interpretan como hora local registrada y los cambios de reloj pueden afectar duraciones derivadas.

## 11. Modelo baseline y resultado ejecutado

Se calcularon media y mediana con **4 225 076 viajes de enero-febrero**. Las constantes aprendidas fueron 25.3567% y 26.3758%.

La regresión lineal se entrenó sobre una muestra aleatoria del 5% por mes de entrenamiento (**211 254 viajes**). Usa imputación, `log1p` de variables numéricas no negativas, escalado y one-hot de categorías. Todas las transformaciones se ajustan en entrenamiento. Se recortan predicciones negativas a cero por coherencia con la definición del objetivo.

Los tres modelos se evaluaron sobre **2 565 402 viajes de marzo**:

| Modelo | Filas de ajuste | MAE (pp) | RMSE (pp) | R² |
| --- | --- | --- | --- | --- |
| Media | 4 225 076 | 9.7479 | 107.6096 | -0.000006 |
| Mediana | 4 225 076 | 9.6848 | 107.6168 | -0.000140 |
| Lineal_log1p | 211 254 | 7.5237 | 107.2835 | 0.006047 |

La regresión reduce el MAE en 22.31% frente a la mediana en esta validación. Es evidencia inicial de señal predictiva; no acredita rendimiento final ni robustez ante cambios de población. Los modelos no tienen igual presupuesto de entrenamiento: los constantes usan todos los objetivos y la regresión usa el 5%; se declara esa diferencia y se evaluará al ampliar el modelado.

La RMSE sigue cerca de 107 pp por observaciones extremas. En marzo hay 12 viajes elegibles con tarifa positiva menor de USD 1. Al informar aparte el segmento con tarifa de al menos USD 1, la RMSE lineal pasa a 24.0122 pp. Este diagnóstico **no sustituye** la evaluación principal, no modifica los modelos y no autoriza a eliminar registros para mejorar una nota.

## 12. Riesgos técnicos, sesgos y limitaciones

- Volumen: leer y concatenar los cuatro meses completos puede consumir mucha memoria. Se procesa cada mes por separado; para equipos con menos recursos se deberá adaptar a lotes manteniendo el mismo diseño y reportando diferencias.
- Faltantes: cinco columnas tienen 27.5983% de faltantes en los registros originales de enero-marzo. Su incidencia en el universo original no equivale a la de la población de tarjeta.
- Anomalías: se hallaron 42 inicios fuera del mes del archivo, incluyendo una fecha de 2008 en marzo. También hay tarifas no positivas y duraciones no positivas.
- Cociente inestable: una tarifa muy pequeña produce porcentajes enormes. En febrero se observa un máximo de 450 000%. Se conservan los extremos en las métricas principales y se revisará su origen.
- Sesgo de observación: efectivo no está observado en `tip_amount`; incluso con tarjeta podría entregarse efectivo adicional. El objetivo es propina registrada, no propina total real.
- Muestreo: usar 5% conserva aproximadamente la mezcla mensual, pero puede omitir casos raros y cambiar el ajuste de una regresión sensible a extremos.
- Temporalidad y representatividad: cuatro meses de Nueva York no representan todo un año ni otras ciudades. La preferencia del pasajero y la calidad del servicio no se observan directamente.
- Uso responsable: se reportan errores por segmentos; las asociaciones no establecen causalidad ni justifican discriminar personas o zonas.

## 13. Plan de trabajo de las semanas restantes

| Semanas | Actividad | Evidencia esperada |
|---|---|---|
| 9 | Revisar anomalías y diccionario; confirmar disponibilidad al predecir | Reglas de limpieza justificadas y conteos |
| 10 | Features y validación cronológica interna | `02_limpieza_features.ipynb`, pipelines y cortes documentados |
| 11 | Comparar al menos tres familias: lineales/robustos, árboles o Random Forest, gradient boosting | Modelos comparables, semillas y resultados guardados |
| 12 | Ajustar hiperparámetros con cortes internos de entrenamiento; comparar en marzo | Tabla de experimentos y justificación de selección |
| 13 | Analizar errores por hora, zona, tarifa y presencia de propina | `04_analisis_errores.ipynb`, tamaños por segmento e interpretación |
| 14 | Congelar decisiones, reentrenar enero-marzo, evaluar una vez en el test reservado | Métricas finales y revisión de limitaciones |
| 15 | Redactar informe final y preparar presentación | PDF/Markdown final y diapositivas |
| 16 | Verificar reproducción completa y preparar defensa | Repositorio final, resultados y exposición |



## Referencias

New York City Taxi & Limousine Commission. (s. f.). *TLC trip record data*. Recuperado el 23 de septiembre de 2026, de https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

New York City Taxi & Limousine Commission. (2025, 18 de marzo). *Data dictionary: Yellow taxi trip records*. https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

City of New York. (s. f.). *Terms of use*. https://www.nyc.gov/main/terms-of-use


