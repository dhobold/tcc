from matplotlib import pyplot as plt, table
import numpy as np
import pandas as pd
from scipy import stats
import Treatment as tr
import Charts as ch
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from scipy.stats import f_oneway

#Definir local que serão salvo as imagens
folder = "c:\\ftp\\Reports\\Images\\"



def Do(aux_df,nome_var):     
    nome_variavel = nome_var
    df = aux_df

    #Remover outliers
    df = tr.remove_outlier(aux_df,'PV')
    
    #Criar tabela padrão de resultados
    if df.empty:
        count = 0
        mean = 0
        mode = 0
        stdDev = 0
        r0 = 0
        r1 = 0
        r2 = 0
        r3 = 0
        r4 = 0
        return nome_variavel,count,mean,mode,stdDev,r0,r1,r2,r3,r4
    
    #Indentificando os Clusters
    df,list_cluster = createClusters(df,15)
    
    #Reduzir número de clusters
    df,list_cluster = reduceClusters(df,list_cluster)
    
    #Gráficos principais para relatório
    ch.timeLine(df,nome_var)  
    
    ch.timelineOficial3(df,nome_var)    
    ch.histograma(df,nome_var)
    ch.boxplot(df,nome_var)   
    
        


    #Graficos Auxiliares
    #ch.heatmap2(df,nome_var,list_cluster)
    #ch.fatoresAVG(df,nome_var)
    #ch.fatoresSTD(df,nome_var)
    #ch.timelineOficial2(df,nome_var)
    #ch.desvPadMovel(df,nome_var)
    #ch.Fourier(df,nome_var)


    #Estatistica básica para gerar o SCORE
    count = df['PV'].count() #Quantidade de elementos
    mean = df['PV'].mean() #Média geral
    mode = df['PV'].mode()[0]    #Moda, elemento que mais se repete
    stdDev = df['PV'].std() #Desvio padrão
    
    
    df = tr.normalizar_clusters(df)
    r0 = abs(mean-mode) #R0 = diferença entre média e moda
    r1 = stdDev #R1 - desvio padrão
    
    stats = df.groupby('CLUSTER')['PV'].agg(['mean', 'std'])
    stats = stats.round(2)
    r2 = stats['std'].mean()
    r3 = stats['mean'].max()/stats['mean'].min()
    
    #Calcular desvio padrão proporcinal dando pesos maior para os clusters mais recentes
    i = 1
    j = 1
    calc = 1
    for i in list_cluster:
        calc = df[df['CLUSTER']==i]['PV'].std() * i
        j = j + i
    r4 = calc/j

        
    
    
    
    plt.close('all')
    return nome_variavel,count,mean,mode,stdDev,r0,r1,r2,r3,r4

def doTable(df):        
    from PIL import Image, ImageDraw, ImageFont
    # Converta o DataFrame para uma string formatada
    table_str = df.to_string(index=False)

    # Defina o tamanho da imagem
    font = ImageFont.load_default()
    lines = table_str.split('\n')
    width = 1000
    height = 500
    image = Image.new('RGB', (width + 20, height + 20), 'white')
    draw = ImageDraw.Draw(image)

    # Desenhe a tabela na imagem
    draw.text((10, 10), table_str, fill='black', font=font)

    # Salve a imagem
    image.save(folder+'table.jpg')

def createClusters(df,w):
    
    #Duplicar coluna datahora
    df['DataHora'] = df['datahora']
    
    #Setar coluna datahora como index
    df.set_index('DataHora', inplace=True)
    
    #Ordernar dataframe por dahora de forma ascendente
    df = df.sort_values(by=['datahora'], ascending=True)

    #Calcular desvio padrao movel
    df['MOVE_STD'] = df['PV'].rolling(window=w).std()
    #Calcula a diferença entre a desvio padrao movel atual em relação a anterior
    df['DIFF_STD'] = abs(df['MOVE_STD'].diff() )
    #Calcula em valores percentuais o quanto variou
    df['DIFF_STD_%'] = abs(df['MOVE_STD'].diff() /df['MOVE_STD'])
    #Calcular média movel
    df['MOVE_AVG'] = df['PV'].rolling(window=w).mean()
    #Calcula a diferença entre a média movel atual em relação a anterior
    df['DIFF_AVG'] = abs(df['MOVE_AVG'].diff() )
    #Calcula em valores percentuais o quanto variou
    df['DIFF_AVG_%'] = abs(df['MOVE_AVG'].diff() /df['MOVE_AVG'])
    
    #Calcular valor para classificar os clusters com base no desvio padrao e na media
    fator_std = (df['DIFF_STD'].mean() + 2*df['DIFF_STD'].std())
    fator_mean = (df['DIFF_AVG'].mean() + 2*df['DIFF_AVG'].std())  

    df['CHANGE_STD'] = df['DIFF_STD'] / fator_std
    df['CHANGE_AVG'] = df['DIFF_AVG'] / fator_mean
    df['CLUSTER'] = 1

    #Verificar as maiores varia~]oes
    aux = df.sort_values(by=['CHANGE_STD'], ascending=False)
    #print(aux.head(10))

    
    #Linhas de segmentação       
    aux = df[(df['CHANGE_STD'] > 1) | (df['CHANGE_AVG'] > 1)]
    aux['DIFF_TIME'] = aux['datahora'].diff()
    #print(aux.head(30))

    #Ordernar dataframe
    aux = aux.sort_values(by=['datahora'], ascending=True)
    df['datahora'] = pd.to_datetime(df['datahora'])
    aux = aux[(aux['DIFF_TIME'] >= pd.Timedelta(minutes=10)) | (aux['DIFF_TIME'].isna() == True)]
    #print(aux)    

    #Criar os clusters
    cluster = 1
    list_cluster = []
    for i in aux.index:
        #verificar hora do inicio do cluster
        momento = pd.to_datetime(aux.loc[i,'datahora'])                      
        #assumir que todo mundo pertence aquele cluster a partir dessa datahora
        df.loc[df['datahora']>=momento,'CLUSTER'] = cluster
        #adicionar cluster na lista para uso futuro
        list_cluster.append(cluster)
        cluster += 1    
    return df,list_cluster
    
def reduceClusters(df,list_cluster):
    #Para cada cluster identificado, realizar o teste ANOVA para verificar se são realmente diferentes    
    for y in range(2):
        j=len(list_cluster)
        z = 0
        
        #Enqunato houver clusters na lista
        while j>1:
            i = 0+z
            #print('Clusters:',list_cluster)        
            
            #Montar o conjunto de dados A
            Ax = list_cluster[i]
            A = df[df['CLUSTER']==Ax]['PV']
            #Montar o conjunto de dados B
            Bx = list_cluster[i+1]        
            B = df[df['CLUSTER']==Bx]['PV']      
            #Executar teste ANOVA para comparar os dois conjuntos de dados  
            #f,p = f_oneway(A,B,nan_policy = 'omit')
            iguais = False
            f = f1 = p = p1 = 1
            if A.nunique()==1 or B.nunique()==1:
                if len(A)==len(B) and A.std() == B.std() and A.mean() == B.mean():
                    iguais = True
            else:
                if iguais == False:
                    f,p = stats.kruskal(A,B)
                    f1,p1 = stats.fligner(A, B)

            tamBx = len(df[df['CLUSTER']==Bx])            
            #H0 -> média A - média B = 0    
            #se o p_value > 0.05 entao significa que não evidencias suficientes para reprovar H0    
            if (p >0.05 and p1 > 0.05) or iguais == True:
                df.loc[df['CLUSTER']==Bx,'CLUSTER'] = Ax
                list_cluster.remove(Bx)
                #print('Cluster A = ',Ax,'| Cluster B = ',Bx,' | kruskal p_value',round(p,3),'| fligner p_value = ',round(p1,3),"| TamBx:",tamBx," ----> são iguais")
            else:
                z = z +1
                #print('Cluster A = ',Ax,'| Cluster B = ',Bx,' | kruskal p_value',round(p,3),'| fligner p_value = ',round(p1,3),"| TamBx:",tamBx)
            j = j-1
            #print('J',j,'Z',z)
            i = i + 1        
    #df.to_csv('c:\\ftp\\df_cluster_reduced.csv',sep=';',decimal=',')    
    return df,list_cluster
    
def teste_normalidade(df,cluster):    
    # Teste de Shapiro-Wilk
    stat, p = stats.shapiro(df['PV'])
    print('Estatística=%.3f, p=%.3f' % (stat, p))

    # Interpretação
    alpha = 0.05
    if p > alpha:
        print('A amostra parece normal (não rejeita H0)',cluster)
    else:
        print('A amostra não parece normal (rejeita H0)',cluster)