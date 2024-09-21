
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def Do(big_df):
    print("Inicio do tratamento dos dados.")
    #Preencher nulos com NA
    big_df.fillna("NA",inplace=True)

    # Convert the date column to date format
    big_df["datahora"] = pd.to_datetime(big_df["datahora"])

    #Criar uma coliuna com a data
    big_df['data'] = big_df['datahora'].dt.date

    #Criar uma coluna com a hora
    big_df['hora'] = big_df['datahora'].dt.hour

    #Criar uma coluna com a data e a hora
    big_df['_dataHora'] = big_df['datahora'].dt.strftime('%Y-%m-%d %H:00')


    #Criar uma coluna com o minuto
    big_df['minuto'] = big_df['datahora'].dt.minute

    #Passar valores para maiuscula
    big_df['nivel_1'] = big_df['nivel_1'].str.upper()
    big_df['nivel_2'] = big_df['nivel_2'].str.upper()
    big_df['nivel_3'] = big_df['nivel_3'].str.upper()
    big_df['nivel_4'] = big_df['nivel_4'].str.upper()
    big_df['tipo'] = big_df['tipo'].str.upper()
    big_df['unidade'] = big_df['unidade'].str.upper()

    #Criar uma coluna com o nome da variavel
    big_df['variavel'] = big_df['nivel_1'] +'-'+ big_df['nivel_2']+'-'+ big_df['nivel_3']+'-'+ big_df['nivel_4']+'-'+ big_df['tipo']+'-'+ big_df['unidade']
    big_df['variavel'] = big_df['variavel'].str.replace("/","_")

    #Gerar um DF com todas as variaveis
    df_var = big_df
    df_var = df_var.drop_duplicates(subset='variavel')

    df_var['count'] = pd.Series(dtype='float')
    df_var['mean'] = pd.Series(dtype='float')
    df_var['mode'] = pd.Series(dtype='float')
    df_var['stdDev'] = pd.Series(dtype='float')
    df_var['r0'] = pd.Series(dtype='float')
    df_var['r1'] = pd.Series(dtype='float')
    df_var['r2'] = pd.Series(dtype='float')
    df_var['r3'] = pd.Series(dtype='float')
    df_var['r4'] = pd.Series(dtype='float')
    df_var['Score'] = pd.Series(dtype='float')
    df_var['done'] = pd.Series(dtype='float')
    
    
    print("Fim do tratamento dos dados.")    
    #Retornar DataFrame
    return big_df,df_var

def cycle(df):
  #Se for variavel de ciclo, entao ignorar ciclos maiores que: 
    # L#1 a L#7  = 18 >= ciclo =< 35
    # L#8        = 12 >= ciclo =< 25
    # L#9        = 22 >= ciclo =< 50
    nome_variavel = df['variavel']
    if 'Ciclo' in nome_variavel:
        if 'Linha 9' in nome_variavel:
            df = df[(df['PV']>=22) & (df['PV']<=50 )]
        else:
            if 'Linha 8' in nome_variavel:
                df = df[(df['PV']>=12) & (df['PV']<=25 )]
            else:
                df = df[(df['PV']>=18) & (df['PV']<=35 )]
    
    return df  

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

# Função para normalizar os dados de cada grupo
def normalizar_clusters(df):
    scaler = MinMaxScaler()
    df['PV'] = scaler.fit_transform(df[['PV']])
    return df

# Função para padronizar os dados de cada grupo
def standardize_group(df):
    scaler = StandardScaler()
    df['PV'] = scaler.fit_transform(df[['PV']])
    return df