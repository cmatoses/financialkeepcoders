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