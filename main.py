import pandas as pd

data_enero = 'Data\yellow_tripdata_2026-01.parquet'

data_febrero = 'Data\yellow_tripdata_2026-01.parquet'

data_marzo = 'Data\yellow_tripdata_2026-01.parquet'

data_abril = 'Data\yellow_tripdata_2026-01.parquet'

#Lectura de datos:

df_enero = pd.read_parquet(data_enero, engine='pyarrow')

df_febrero = pd.read_parquet(data_febrero, engine='pyarrow')

df_marzo = pd.read_parquet(data_marzo, engine='pyarrow')

df_abril = pd.read_parquet(data_abril, engine='pyarrow')