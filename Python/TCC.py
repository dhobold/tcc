from datetime import datetime
import numpy as np
import pandas as pd

import Ingestion as ingestion
import Treatment as treatment
import Analyzing as analyzing
import Reporting as reporting

pd.options.mode.chained_assignment = None 

inicioProcesso = datetime.now()
print("#################################################")
print("Inicio")

#Realizar a ingestão dos dados
big_df = ingestion.Do()

#Realizar o tratamento dos dados
big_df,df_var = treatment.Do(big_df)

#Especificar quais variaveis queremos gerar o relatorio
#df_var = df_var[df_var['nivel_1']=='LINHA 1']
df_var = df_var[df_var['nivel_2']=='SELADORA']
#df_var = df_var[df_var['nivel_3']=='FLAUTA']
#df_var = df_var[df_var['nivel_4']=='TERMOPAR']
df_var = df_var[df_var['tipo']=='TEMPERATURA']


#Realizar a analise dos dados
aux_var = analyzing.Do(df_var,big_df)

#Realizar a criação do relatório
reporting.Do(aux_var)


########################################
#Plotar DataFrame com analise das anormalidades
aux_var.to_csv('c:\\ftp\\out.csv',sep=';',decimal=',')

print(aux_var[['variavel','count','mean','mode','r0','r1','r2','r3','r4','Score']].head(10))

########################################
fimProcesso = datetime.now()
tempoTotal = (fimProcesso - inicioProcesso).total_seconds()
print("Tempo total de processo: "+ str(tempoTotal)+" segundos")
print("Fim!!!")


