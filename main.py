import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_enero = 'Data\yellow_tripdata_2026-01.parquet'
data_febrero = 'Data\yellow_tripdata_2026-02.parquet'
data_marzo = 'Data\yellow_tripdata_2026-03.parquet'
data_abril = 'Data\yellow_tripdata_2026-04.parquet'

#Lectura de datos:

df_enero = pd.read_parquet(data_enero, engine='pyarrow')
df_febrero = pd.read_parquet(data_febrero, engine='pyarrow')
df_marzo = pd.read_parquet(data_marzo, engine='pyarrow')
df_abril = pd.read_parquet(data_abril, engine='pyarrow')


#Exploracion de datasets:
# Se verifica si contienen exactamente las mismas variables
print(df_enero.columns.equals(df_febrero.columns))
print(df_enero.columns.equals(df_marzo.columns))
print(df_enero.columns.equals(df_abril.columns))


# Funcion para obtener información básica de cada dataset (tipo, cant. valores faltantes y su porcentaje)
def explorar_dataset(df, nombre):
    print(f"\n===== {nombre} =====")

    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    resumen = pd.DataFrame({
        "Tipo de dato": df.dtypes,
        "Valores faltantes": df.isnull().sum(),
        "Faltantes (%)": df.isnull().mean() * 100
    })

    print("\nResumen de variables:")
    print(resumen)

def revisar_objetivo(df, nombre):
    print(f"\n===== {nombre} =====")

    print("Total de viajes:", len(df))

    print("Viajes con tarjeta:",(df["payment_type"] == 1).sum())

    print("Fare <= 0:",(df["fare_amount"] <= 0).sum())

    print("Tip = 0:",(df["tip_amount"] == 0).sum())

explorar_dataset(df_enero, "Enero")

explorar_dataset(df_febrero, "Febrero")

explorar_dataset(df_marzo, "Marzo")

explorar_dataset(df_abril, "Abril")



# Revisión de variables relacionadas con el objetivo:
# Se analiza la cantidad de viajes pagados con tarjeta, los viajes sin
# propina registrada y los registros con una tarifa menor o igual a cero.
# Esta revisión permite determinar qué observaciones son adecuadas para
# calcular el porcentaje de propina.

revisar_objetivo(df_enero, "Enero")
revisar_objetivo(df_febrero, "Febrero")
revisar_objetivo(df_marzo, "Marzo")
revisar_objetivo(df_abril, "Abril")

# Relación entre método de pago y ausencia de propina registrada:
# Se analiza la cantidad de registros con tip_amount igual a cero
# según el método de pago. Esto es relevante debido a que las propinas
# en efectivo no se registran en tip_amount.

print("===== ENERO =====")
print(pd.crosstab(
    df_enero["payment_type"],
    df_enero["tip_amount"] == 0
))

print("\n===== FEBRERO =====")
print(pd.crosstab(
    df_febrero["payment_type"],
    df_febrero["tip_amount"] == 0
))

print("\n===== MARZO =====")
print(pd.crosstab(
    df_marzo["payment_type"],
    df_marzo["tip_amount"] == 0
))

print("\n===== ABRIL =====")
print(pd.crosstab(
    df_abril["payment_type"],
    df_abril["tip_amount"] == 0
))

print("Enero:", (df_enero["fare_amount"] == 0).sum())
print("Febrero:", (df_febrero["fare_amount"] == 0).sum())
print("Marzo:", (df_marzo["fare_amount"] == 0).sum())
print("Abril:", (df_abril["fare_amount"] == 0).sum())

print("Enero:", (df_enero["fare_amount"] < 0).sum())
print("Febrero:", (df_febrero["fare_amount"] < 0).sum())
print("Marzo:", (df_marzo["fare_amount"] < 0).sum())
print("Abril:", (df_abril["fare_amount"] < 0).sum())


# Juntar los 4 datasets en 1 solo
df_completo = pd.concat(
    [df_enero, df_febrero, df_marzo, df_abril],
    ignore_index=True
)

# Nueva variable que nos da el mes del viaje a partir de la fecha de recogida
# para identificar el mes correspondiente de cada viaje
df_completo["month"] = df_completo["tpep_pickup_datetime"].dt.month
print(df_completo.shape)

# Copia del dataset con los viajes que nos interesan
# Se han seleccionado los viajes pagados con tarjeta y con tarifa psotiva
# para crear el target "tip_percentage" 

df_modelo = df_completo[
    (df_completo["payment_type"] == 1) &
    (df_completo["fare_amount"] > 0)
].copy()


df_modelo["trip_duration"] = (
    df_modelo["tpep_dropoff_datetime"] -
    df_modelo["tpep_pickup_datetime"]
).dt.total_seconds() / 60

df_modelo["tip_percentage"] = (
    df_modelo["tip_amount"] /
    df_modelo["fare_amount"]
) * 100

# Datos estadisitcos del dataset
print(df_modelo["tip_percentage"].describe())

# Definicion de datos de entrenamiento (75%) aprox y de prueba (25%) aprox

df_train = df_modelo[df_modelo["month"] <= 3].copy()
df_test = df_modelo[df_modelo["month"] == 4].copy()

# Baseline: promedio del porcentaje de propina del conjunto de entrenamiento
# Se utiliza el promedio del porcentaje de propina del conjunto de
# entrenamiento como predicción para todos los viajes del conjunto de prueba.
baseline = df_train["tip_percentage"].mean()

print("Predicción del baseline:", baseline)

y_test = df_test["tip_percentage"]

y_pred_baseline = [baseline] * len(df_test)

mae = mean_absolute_error(y_test, y_pred_baseline)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
r2 = r2_score(y_test, y_pred_baseline)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
