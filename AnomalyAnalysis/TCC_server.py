import argparse
from datetime import datetime
import json
import os
import shutil
import numpy as np
import pandas as pd

import Common as cm
import Ingestion as ingestion
import Treatment as treatment
import Analyzing as analyzing
import Reporting as reporting

pd.options.mode.chained_assignment = None 

# pega os parametros de execucao
parser = argparse.ArgumentParser(description="Executar analise de dados local ou como servidor.")
parser.add_argument('--type', type=str, required=True, help='cmd ou server')
args = parser.parse_args()
_cmd = True if args.type == "cmd" else False

_df_var = pd.DataFrame()
_big_df = pd.DataFrame()
_aux_var = pd.DataFrame()


def start_server():
    global _df_var 
    global _big_df
    global _aux_var
    try:
        _big_df = ingestion.Do()
        _big_df, _df_var = treatment.Do(_big_df)
        return f"Ingestão e tratamento de {_big_df.shape[0]} dados finalizados"
    except Exception as e:  
        return "Ocorreu um erro ao carregar os dados" 

def UniqueValuesToJson(df, colunas):
    """
    Retorna um dicionário contendo os valores únicos de cada coluna especificada no DataFrame.
    """
    resultado_dict = {}
    
    # Itera sobre a lista de colunas e coleta valores únicos
    for coluna in colunas:
        if coluna in df.columns:
            valores_unicos = df[coluna].unique().tolist()
            resultado_dict[coluna] = valores_unicos
    
    return resultado_dict  # Retorna um dicionário Python com múltiplas colunas

def LoadList():
    try:
        # Inclua as colunas que você deseja retornar
        return UniqueValuesToJson(_df_var, ['nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'tipo'])
    except Exception as e:
        return {}
    
def setVariables(data):
    global _df_var

    nivel_1 = data.get('nivel_1')
    nivel_2 = data.get('nivel_2')
    nivel_3 = data.get('nivel_3')
    nivel_4 = data.get('nivel_4')
    tipo = data.get('tipo')

    # TODO: Filtar em df copia,para noa perder o principal para nova analise

    # Verifica e aplica filtros apenas para variáveis que não são vazias
    if nivel_1:
        _df_var = _df_var[_df_var['nivel_1'] == f"{nivel_1}"]
    if nivel_2:
        _df_var = _df_var[_df_var['nivel_2'] == f"{nivel_2}"]
    if nivel_3:
        _df_var = _df_var[_df_var['nivel_3'] == f"{nivel_3}"]
    if nivel_4:
        _df_var = _df_var[_df_var['nivel_4'] == f"{nivel_4}"]
    if tipo:
        _df_var = _df_var[_df_var['tipo'] == f"{tipo}"]

def AnalyzingData(data):
    global _aux_var
    global _df_var

    try:
        cm.deleteAllFilesInFolder(cm.path_images)
        setVariables(data)
        _aux_var = analyzing.Do(_df_var, _big_df)
    except Exception as e:
        print(f"Erro: {e.args[0]}")
        

def dataAnalysis():
    return _aux_var

def reportingData():
    global _aux_var
    reporting.Do(_aux_var)

def copiar_arquivo():
    pdf_filename = 'relatorio.pdf'
    source_path = os.path.join('Reports', pdf_filename)
    destination_path = os.path.join('AnomalyAnalysis','static')
    
    if os.path.exists(source_path):
        # Mover o arquivo para a pasta static
        shutil.copy(source_path, destination_path)
        print(f"Arquivo movido para {destination_path}")
    else:
        print("Arquivo não encontrado em Reports")



if _cmd:
    start_server()

