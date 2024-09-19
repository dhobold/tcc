from datetime import datetime
import warnings
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import Ingestion as ingestion
import Treatment as treatment
import Analyzing as analyzing
import Reporting as reporting


#Criar pastas para armazenar as imagens e relatório
folder = Path("c:\\ftp\\Reports\\Images")
if not folder.exists():
    folder.mkdir(parents=True, exist_ok=True)

#Desabilitar warning
pd.options.mode.chained_assignment = None 
warnings.filterwarnings("ignore", category=UserWarning, message=".*FixedLocator.*")


#Iniciar processo
inicioProcesso = datetime.now()
print("#################################################")
print("Inicio")

#Realizar a ingestão dos dados
big_df = ingestion.Do()

#Realizar o tratamento dos dados
big_df,df_var = treatment.Do(big_df)


#Especificar quais variaveis queremos gerar o relatorio
#df_var = df_var[df_var['nivel_1']=='LINHA 2']
#df_var = df_var[df_var['nivel_2']=='COS']
#df_var = df_var[df_var['nivel_3']=='TEMPO CT']
#df_var = df_var[df_var['nivel_4']=='CENTRO ESQUERDO']
#df_var = df_var[df_var['nivel_4']=='SUPERIOR ESQUERDO']
df_var = df_var[df_var['tipo']=='TEMPERATURA']

#Realizar a analise dos dados
aux_var = analyzing.Do(df_var,big_df)

#Realizar a criação do relatório
reporting.Do(aux_var)

########################################
#Plotar DataFrame com analise das anormalidades
print(aux_var[['variavel','count','mean','mode','r0','r1','r2','r3','r4','Score']].head(10))

########################################
fimProcesso = datetime.now()
tempoTotal = (fimProcesso - inicioProcesso).total_seconds()
print("Tempo total de processo: "+ str(tempoTotal)+" segundos")
print("Fim!!!")


