import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


#Importar dados da pasta
path = 'C:\\Users\\pohed\\OneDrive\\DIH\\Pós FACENS\\TCC\\Ingestion'
# Get CSV files list from a folder
csv_files = glob.glob(path + "/*.csv")

# Read each CSV file into DataFrame
# This creates a list of dataframes
df_list = (pd.read_csv(file,delimiter=';',decimal=',') for file in csv_files)

# Concatenate all DataFrames
big_df   = pd.concat(df_list, ignore_index=True)

#Conhecendo os dados
print(big_df.info())

#Qual formado da tabela?
print("Formato da tabela")
print(big_df.shape)

#Visualizando as primeiras 10 informações
print(big_df.head(10))

#Verificar dados duplicados
print("Quantos dados únicos cada coluna possui?")
print(big_df.nunique())

#Verificar quantas variaives existem

big_df['variavel'] = big_df['nivel_1'] +'-'+ big_df['nivel_2']+'-'+ big_df['nivel_3']+'-'+ big_df['nivel_4']+'-'+ big_df['tipo']+'-'+ big_df['unidade']


print('################################################')
print('Quantas variaveis existem na base de dados?')
print(big_df['variavel'].nunique())

print('################################################')
print('Quantos tipos de dados existem na base de dados?')
print(big_df['tipo'].nunique())

print('################################################')
print('Quais são eles?')

print(big_df['tipo'].unique().tolist())


print('################################################')
print('Como as variaveis são divididadas?')
print('################################################')
print("Nivel 1")
print(big_df['nivel_1'].unique().tolist())

print('################################################')
print("Nivel 2")
print(big_df['nivel_2'].unique().tolist())

print('################################################')
print("Nivel 3")
print(big_df['nivel_3'].unique().tolist())

print('################################################')
print("Nivel 4")
print(big_df['nivel_4'].unique().tolist())


print('################################################')
print("Unidade de Medida")
print(big_df['unidade'].unique().tolist())

#################################################################
#Um gráfico de exemplo dos dados
def graficos(aux_df,n1,n2,n3,n4,tp,un):
    df = aux_df
    df = df[df['nivel_1']==n1]
    print(df['variavel'].unique().tolist())
    df = df[df['nivel_2']==n2]
    print(df['variavel'].unique().tolist())
    df = df[df['nivel_3']==n3]
    print(df['variavel'].unique().tolist())
    df = df[df['nivel_4']==n4]
    print(df['variavel'].unique().tolist())
    df = df[df['tipo']==tp]
    print(df['variavel'].unique().tolist())
    df = df[df['unidade']==un]
    print(df['variavel'].unique().tolist())

    print(df.describe())

    #Linha do tempo
    df['PV'].plot()
    plt.title(n1+'-'+n2+'-'+n3+'-'+n4+'-'+tp+'-'+un)
    plt.savefig('c:\\ftp\\timeline', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    #Calcular desvio padrão
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    min = mean -3*std_dev
    max = mean + 3*std_dev

    #Eliminar dados que estão além de 3 desvios
    df = df[(df['PV']>=min) & (df['PV']<= max)]
    
    #Plot histograma
    plt.hist(df['PV'],bins=300)
    plt.title(n1+'-'+n2+'-'+n3+'-'+n4+'-'+tp+'-'+un)
    # Save the plot as a PNG
    plt.savefig('c:\\ftp\\histograma', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

graficos(big_df,'Linha 2','COS','BOMBA','Termopar','Temperatura','Celsius')


