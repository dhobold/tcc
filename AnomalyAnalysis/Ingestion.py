
from datetime import datetime
import glob
import pandas as pd

def Do():
    
    print("Inicio da Ingestão dos dados.")
    #Importar dados da pasta
    path = 'Ingestion'
    # Get CSV files list from a folder
    csv_files = glob.glob(path + "/*.csv")

    #Ler cada arquivo CSV na pasta
    df_list = (pd.read_csv(file,delimiter=';',decimal=',') for file in csv_files)

    # Concatenate all DataFrames
    big_df   = pd.concat(df_list, ignore_index=True)
    big_df['PV'] = big_df['PV']
    print("Fim da Ingestão dos dados.")
    #Retornar DataFrame
    return big_df