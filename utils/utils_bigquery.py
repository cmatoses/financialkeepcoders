from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
import os
import pandas as pd


# Ruta al archivo de credenciales JSON
key_path = "conf/bigquery_credentials.json"

# ID del proyecto y del dataset
project_id = 'sara-carles-keepcoding'

# Datasets
datasets =  {
        'gold': 'sara-carles-keepcoding.gold',
        'silver': 'sara-carles-keepcoding.silver',
        'bronze': 'sara-carles-keepcoding.bronze'
            }


"""BigQuery Utils"""
class BigQueryUtils:
    """Class with methods to interact with the BigQuery"""
    def __init__(self, key_path)-> None:
        self.key_path=key_path
        self.credentials = service_account.Credentials.from_service_account_file(key_path)
        self.client = bigquery.Client(credentials=self.credentials)

        
    def run_query(self, query:str):
        """"""
        query_job = self.client.query(query)
        
        return query_job.to_dataframe()
    

    def save_dataframe(self, df:pd.DataFrame, project_id:str, dataset:str, table:str, if_exists, schema)-> None:
        """Function to save and create the table if it's necessary to load data in BigQuery

        Parameters:
        - df (pd.DataFrame): Pandas DataFrame with the data to save in BigQuery
        - dataset (str): BigQuery dataset when it has to be stored the data.
        - table (str): BigQuery table when it has to be stored the data.
        Returns:
        - None
        """
        table_id = f'{project_id}.{dataset}.{table}'
        if schema:
            pd.DataFrame.to_gbq(
                df,
                destination_table=table_id,
                project_id=project_id,
                if_exists=if_exists,
                credentials=self.credentials,
                table_schema= schema,
                chunksize = 10000
            )
        else: 
                pd.DataFrame.to_gbq(
                df,
                destination_table=table_id,
                project_id=project_id,
                if_exists=if_exists,
                credentials=self.credentials,
                chunksize = 10000
            )
                
# Obtener todos los tickers de los distintos mercados
def get_unique_tickers():
    """
    Obtener todos los tickers de los distintos mercados

    Returns:
        Dataframe con la columna tickers y sus valores
    """
    import yahoo_fin.stock_info as si

    # Obtener todos los tickers de los índices especificados
    tickers_dow = si.tickers_dow()
    #tickers_sp500 = si.tickers_sp500()
    #tickers_nasdaq = si.tickers_nasdaq()
    
    # Combinar todos los tickers en una sola lista
    # combined_tickers = tickers_dow + tickers_sp500 + tickers_nasdaq
    
    # Eliminar duplicados pero mantener el primer encuentro
    # unique_tickers = list(dict.fromkeys(combined_tickers))
    unique_tickers = list(dict.fromkeys(tickers_dow))

    
    # Crear un DataFrame con los tickers únicos
    unique_tickers = pd.DataFrame(unique_tickers, columns=['ticker'])
    
    return unique_tickers


def initial_date_by_ticker(unique_tickers, initial_date='2015-01-01'):
    """
    Añade una columna de fecha inicial a un DataFrame de tickers.

    Parameters:
    - unique_tickers (pd.DataFrame): DataFrame que contiene una columna 'ticker' con los tickers únicos.
    - initial_date (str): Fecha inicial en formato 'YYYY-MM-DD' para agregar al DataFrame. Por defecto es '2015-01-01'.
    
    Returns:
    - pd.DataFrame: DataFrame original con una nueva columna llamada 'initial_date', que contiene la fecha inicial proporcionada.
    """
    # Importamos las librerias
    import pandas as pd
    from datetime import datetime

    # Convertir la fecha inicial a un objeto datetime
    initial_date = datetime.strptime(initial_date, '%Y-%m-%d')
   
    # Añadimos la columna
    unique_tickers['initial_date'] = initial_date
    
    # Convertir todas las fechas a tz-naive (sin zona horaria)
    unique_tickers['initial_date'] = unique_tickers['initial_date'].dt.tz_localize(None)

    return unique_tickers


def last_date_by_ticker_saved(max_date_by_ticker, bigquery, table_conca):
    """
    Obtiene la fecha máxima por ticker a partir de los datos en BigQuery y actualiza el DataFrame de fechas.

    Parameters:
    - max_date_by_ticker: DataFrame que contiene tickers y la fecha inicial.
    - bigquery: Instancia de conexión a BigQuery.
    - table_conca: Nombre de la tabla en BigQuery.

    Returns:
    - DataFrame con la fecha máxima actualizada por ticker.
    """
    # Retrieve data from BigQuery
    import pandas as pd

    current_data = bigquery.run_query(
        f"""
        SELECT
            * 
        FROM {table_conca} 
        """
    )
    
    # Convertir todas las fechas a tz-naive (sin zona horaria)
    current_data['date'] = current_data['date'].dt.tz_localize(None)
    
    # Obtener la fecha máxima por ticker
    query_ticker = current_data.groupby('ticker')['date'].max().reset_index()
    query_ticker['date'] = pd.to_datetime(query_ticker['date'])
    
    # Mergear con el DataFrame de tickers únicos
    max_date_by_ticker = pd.merge(max_date_by_ticker, query_ticker, how='inner', on='ticker')
    max_date_by_ticker['date'] = max_date_by_ticker[['date', 'initial_date']].max(axis=1)
    
    # Eliminar la columna auxiliar 'initial_date'
    max_date_by_ticker.drop(columns=['initial_date'], inplace=True)
    
    return max_date_by_ticker


def fetch_historical_data(max_date_by_ticker, ticker_col='ticker', start_date_col='date', interval='1d'):
    """
    Obtiene datos históricos para cada ticker en el DataFrame proporcionado desde una fecha inicial hasta la fecha actual.

    Parameters:
    - max_date_by_ticker (pd.DataFrame): DataFrame que contiene columnas de tickers y una columna de fechas (por defecto 'date') con la fecha inicial.
    - ticker_col (str): Nombre de la columna en `max_date_by_ticker` que contiene el ticker. Por defecto es 'ticker'.
    - start_date_col (str): Nombre de la columna en `max_date_by_ticker` que contiene la fecha inicial. Por defecto es 'date'.
    - interval (str): Intervalo de los datos históricos (por ejemplo, '1d' para diario). Por defecto es '1d'.
    
    Returns:
    - pd.DataFrame: DataFrame que contiene todos los datos históricos para los tickers.
    """
    import pandas as pd
    from datetime import date
    import yahoo_fin.stock_info as si

    data = []

    for _, row in max_date_by_ticker.iterrows():
        ticker = row[ticker_col]
        start_date = row[start_date_col] + pd.Timedelta(days=1)
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()  # Convertir a datetime de Python

        try:
            # Obtener datos históricos
            data_row = si.get_data(ticker, start_date=start_date, end_date=date.today(), index_as_date=False, interval=interval)
            # Agregar el ticker como una columna en los datos históricos
            data_row[ticker_col] = ticker
            # Añadir los datos a la lista
            data.append(data_row)
            print(f"Datos obtenidos para {ticker}.")
        except Exception as e:
            print(f"Error al obtener datos para {ticker}: {e}")

    # Concatenar todos los DataFrames en uno solo
    df = pd.concat(data, ignore_index=True)
    
    return df



def process_and_save_joins(df, max_date_by_ticker, bigquery, project, dataset, table):
    """
    Realiza LEFT JOIN e INNER JOIN entre dos DataFrames y guarda los resultados en BigQuery.

    Parameters:
    - df (pd.DataFrame): DataFrame principal que contiene los datos a combinar.
    - max_date_by_ticker (pd.DataFrame): DataFrame que contiene los tickers y las fechas de referencia para combinar.
    - bigquery (object): Instancia de conexión a BigQuery para guardar el resultado.
    - project (str): Nombre del proyecto en BigQuery.
    - dataset (str): Nombre del dataset en BigQuery.
    - table (str): Nombre de la tabla en BigQuery donde se guardará el resultado.
    - schema (list of dict): Esquema de la tabla en BigQuery, especificado como una lista de diccionarios.
    
    Returns:
    - None: La función guarda el resultado directamente en BigQuery.
    """
    # Convertir las columnas de fecha a datetime en formato UTC
    max_date_by_ticker['date'] = pd.to_datetime(max_date_by_ticker['date'], utc=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Realizar el LEFT JOIN
    left_join = pd.merge(df, max_date_by_ticker, on=['ticker', 'date'], how='left')

    # Realizar el INNER JOIN
    inner_join = pd.merge(df, max_date_by_ticker, on=['ticker', 'date'], how='inner')

    # Identificar las filas del LEFT JOIN que no están en el INNER JOIN
    left_join_tuples = left_join[['ticker', 'date']].apply(tuple, axis=1)
    inner_join_tuples = inner_join[['ticker', 'date']].apply(tuple, axis=1)

    # Obtener los datos que están en left_join pero no en inner_join
    result = left_join[~left_join_tuples.isin(inner_join_tuples)]

    return result





#   # Creating table in Bigquery
#    def create_table(key_path, project_id, dataset_id, table_id, schema):
#        try:
#            # Configura las credenciales y crea una instancia del cliente BigQuery
#            credentials, client = gcp_auth(key_path, project_id)   
#
#            # Define el nombre completo de la tabla
#            table_ref = client.dataset(dataset_id).table(table_id)
#
#            # Verifica si la tabla ya existe
#            try:
#                client.get_table(table_ref)
#                print(f"La tabla {table_id} ya existe en el dataset {dataset_id}.")
#            except NotFound:
#                # Configura la tabla con su esquema
#                table = bigquery.Table(table_ref, schema=schema)
#                
#                # Crea la tabla en BigQuery
#                table = client.create_table(table)
#                print(f"Tabla {table.table_id} creada en el dataset {dataset_id}.")
#
#        except Exception as e:
#            # Manejo de errores
#            print(f"Error durante la creación de la tabla: {e}")