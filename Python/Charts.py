
from datetime import datetime
from matplotlib import pyplot as plt, table
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy import stats
from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import Treatment as treatment
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from scipy.stats import f_oneway

current_directory = Path(__file__).parent
folder = str(current_directory)+"\\Reports\\Images\\"

######################################################
#Gráfico 1 - Linha do tempo mais 3x desvio padrão
######################################################

def timeLine(df,nome_var):
    nome_variavel = nome_var
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    
    #pv_avg = df['PV'].rolling(window=120).apply(lambda x: x.mode()[0])
    pv_avg = df['PV'].rolling(window=120).mean()

    #Linha do tempo
    fig, ax = plt.subplots()
    ax.plot(df['PV'])
    ax.plot(pv_avg,color='k')
    xmin,xmax = ax.get_xlim()
    
    #Linhas de 1 desvio padrão
    min = mean -1*std_dev
    max = mean + 1*std_dev 
    ax.axhline(y=min,color='g',linestyle='--')
    plt.text(xmax, min, '-1dev')    
    ax.axhline(y=max,color='g',linestyle='--')
    plt.text(xmax, max, '+1dev')
    
    #Linhas de 2 desvio padrão
    min = mean -2*std_dev
    max = mean + 2*std_dev 
    ax.axhline(y=min,color='y',linestyle='--')
    plt.text(xmax, min, '-2dev')    
    ax.axhline(y=max,color='y',linestyle='--')
    plt.text(xmax, max, '+2dev')

    #Linhas de 3 desvio padrão
    min = mean -3*std_dev
    max = mean + 3*std_dev 
    ax.axhline(y=min,color='r',linestyle='--')
    plt.text(xmax, min, '-3dev')    
    ax.axhline(y=max,color='r',linestyle='--')
    plt.text(xmax, max, '+3dev')

    ax.axhline(y=mean,color='g',linestyle='--')
    plt.text(xmax, mean, 'Mean')
    
    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)
    
    
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image1.jpg", format='jpg', dpi=100)
    plt.clf()
    #Calcular quantos pontos estão dentro ou fora de 1 desvio padrao
    min = mean -1*std_dev
    max = mean + 1*std_dev 
    df['r1'] = np.where(df['PV']>max,'NOK','OK')
    df['r1'] = np.where(df['PV']<min,'NOK','OK')
    r = len(df[df['r1']=='NOK']) / len(df['variavel'])
    return r


def desvioPadraoMovel(df,nome_var):
    #Tratar dados de ciclo
    df = treatment.cycle(df)
    mean = df['PV'].mean()
    nome_variavel = nome_var
    #Calcula desvio padrão do dia    
    df_std = df.groupby(['data','hora','variavel'])['PV'].std().reset_index()
    #df_std['_PV'] = df_std['PV']/mean*100
    df_std['_PV'] = df_std['PV']
    #|Analise do desvio padrao
    fig, ax = plt.subplots()
    #ax.set_title("Desvio Padrao / Mean * 100")
    ax.set_title("Desvio Padrao")
    ax.plot(df_std['_PV'])
    z = np.polyfit(df_std.index,df_std['_PV'],1)
    p = np.poly1d(z)
    plt.plot(df_std.index,p(df_std.index),color='r',linestyle='--')
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    fator_a = round(z[0],2)
    fator_b = round(z[1],2)
    texto = ""+str(fator_a)+'x + '+str(fator_b)
    plt.text(xmax*0.8,ymax , texto)
    
    plt.savefig(folder+nome_variavel+"_image2.jpg", format='jpg', dpi=100)
    plt.clf()

    if fator_a < 0:
        r = 0
    else:
        r = fator_a
    return r


def histograma(df,nome_var):
    #Tratar dados de ciclo
    df = treatment.cycle(df)
    
    
    nome_variavel = nome_var
    #Eliminar dados que estão além de 3 desvios
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    
    min = mean -6*std_dev
    max = mean + 6*std_dev  
    df2 = df[(df['PV']>=min) & (df['PV']<= max)]
    
    #Plot histograma
    fig, ax = plt.subplots()
    ax.hist(df2['PV'],bins=60)
    ax.set_title("Histograma")
    
    min = mean -3*std_dev
    max = mean + 3*std_dev    


    ax.axvline(x=min,color='r',linestyle='--')
    ax.axvline(x=max,color='r',linestyle='--')
    ax.set_xlim(mean - 6*std_dev ,mean + 6*std_dev )
    ymin,ymax = ax.get_ylim()
    
    plt.text(min, ymax*0.9, '-3dev')
    plt.text(max, ymax*0.9, '+3dev')
    
    # Save the plot as a PNG
    plt.savefig(folder+nome_variavel+"_image3.jpg", format='jpg', dpi=100)
    plt.clf()
    r =0
    return r

def anormalidade(df,nome_var):
    #Tratar dados de ciclo
    df = treatment.cycle(df)

    
    w = 60
    df['MOVE_STD'] = df['PV'].rolling(window=w).std()
    df['MOVE_AVG'] = df['PV'].rolling(window=w).mean()
    df['DIFF_AVG'] = abs(df['MOVE_AVG'].diff() )
    df['DIFF_STD'] = abs(df['MOVE_STD'].diff() )
    
    df['MOVE_STD'].fillna(df['MOVE_STD'].mean(),inplace=True)

    nome_variavel = nome_var
    #Analise de anormalidade
    df = df.groupby(['data','hora','variavel'])['MOVE_STD'].mean().reset_index()
    #df.groupby(['data','hora','variavel'])['PV'].agg(pd.Series.mode).to_frame()
    
    outliers_fraction = float(.1)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df['MOVE_STD'].to_numpy().reshape(-1,1))
    data = pd.DataFrame(np_scaled)
    #print(data)

    # train isolation forest
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data) 

    df['anomaly'] = model.predict(data)

    # visualization
    #figsize=(10,6)
    fig, ax = plt.subplots()

    a = df.loc[df['anomaly'] == -1, ['MOVE_STD']] #anomaly

    ax.plot(df.index, df['MOVE_STD'], color='black', label = 'Normal')
    ax.scatter(a.index,a['MOVE_STD'], color='red', label = 'Anomaly')
    plt.legend()
    plt.title("Detecção de anormalidade no desvio padrao")
    plt.savefig(folder+nome_variavel+"_image4.jpg", format='jpg', dpi=100)
    #plt.show();
    plt.clf()
    
    r = len(a)
    return r

def timeLineAVG(df,nome_var):
    nome_variavel = nome_var
   
    #Tratar dados de ciclo
    df = treatment.cycle(df)
    
    #Calcular média movel de 60 minutos
    #pv_avg = df['PV'].rolling(window=600).mean()
    pv_avg = df['PV'].rolling(window=60).apply(lambda x: x.mode()[0])
    #Calcular desvio padrão    
    std_dev = pv_avg.std()
    mean = pv_avg.mean()
    
    min = mean -1*std_dev
    max = mean + 1*std_dev    
    #Linha do tempo
    fig, ax = plt.subplots()
    
    ax.plot(pv_avg,color='b')
    ax.axhline(y=min,color='r',linestyle='--')
    ax.axhline(y=max,color='r',linestyle='--')
    ax.axhline(y=mean,color='r',linestyle='--')
    

    xmin,xmax = ax.get_xlim()
    
    plt.text(xmax, min, '-1dev')
    plt.text(xmax, max, '+1dev')
    plt.text(xmax, mean, 'Mean:')    
    #plt.text(xmax, mean, str(round(mean,2)) )  
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image1.jpg", format='jpg', dpi=100)
    plt.clf()
    #Calcular quantos pontos estão dentro ou fora do desvio padrao
    df['r1'] = np.where(df['PV']>max,'NOK','OK')
    df['r1'] = np.where(df['PV']<min,'NOK','OK')
    r = len(df[df['r1']=='NOK']) / len(df['variavel'])
    return r

def timeLine2(df,nome_var):
    nome_variavel = nome_var
    
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    #Calcular média movel
    pv_avg = df['PV'].rolling(window=60).apply(lambda x: x.mode()[0])
    min = mean -3*std_dev
    max = mean + 3*std_dev    
    #Linha do tempo
    fig, ax = plt.subplots()
    ax.plot(df['datahora'],df['PV'])
    ax.plot(pv_avg,color='b')
    ax.axhline(y=min,color='r',linestyle='--')
    ax.axhline(y=max,color='r',linestyle='--')
    ax.axhline(y=mean,color='r',linestyle='--')
    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)

    xmin,xmax = ax.get_xlim()
    
    plt.text(xmax, min, '-3dev')
    plt.text(xmax, max, '+3dev')
    plt.text(xmax, mean, 'Mean')        
    
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image5.jpg", format='jpg', dpi=100)
    plt.clf()
    
    r = 0
    return r

def similaridade(df,nome_var):
    table = pd.pivot_table(df, values='PV', index=['minuto'],
                       columns=['_dataHora'], aggfunc="mean")

    #table.to_csv('c:\\ftp\\table.csv',sep=';',decimal=',')

    #print(len(table.columns))

    
    result = pd.DataFrame(index=np.arange(len(table.columns)), columns=np.arange(len(table.columns)),dtype=float)
    count = 0
    for j in range(len(table.columns)):
        #print('********************')
        for i in range(len(table.columns)):    
            f,p = f_oneway(table.iloc[:,j],table.iloc[:,i],nan_policy = 'omit')
            result.iloc[i,j] = p
            if p <0.05:
                count = count +1
            #print(i,round(p,4))
    #print(result)

    import matplotlib.pyplot as plt
    result.fillna(1,inplace=True)

    result[result<0.05] = 0.
    result[result>=0.05] = 1.

    #print(result.astype())

    fig, ax = plt.subplots()
    im = ax.imshow(result.to_numpy())
    ax.set_xticks(np.arange(len(result.columns)))
    ax.set_yticks(np.arange(len(result.columns)))
    # Create colorbar
    cbarlabel = 'p_value'
    cbar_kw =  {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


    fig.tight_layout()
    
    #plt.show()
    plt.savefig(folder+nome_var+"_image4.jpg", format='jpg', dpi=100)
    aux = table.mean()
    #r = abs((aux.min()/aux.max()-1)*100)
    r = count
    return r

def timelineDESPAD(df,nome_var):
    r = 0
    nome_variavel = nome_var

    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()    
    
    pv_avg = df['PV'].rolling(window=120).std()

    #Linha do tempo
    fig, ax = plt.subplots()
    #ax.plot(df['PV'])
    ax.plot(pv_avg,color='k')
    xmin,xmax = ax.get_xlim()
    
    #Linhas de 1 desvio padrão
    min = mean -1*std_dev
    max = mean + 1*std_dev     
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image11.jpg", format='jpg', dpi=100)
    plt.clf()    
    return r

def timelineDESPAD_2(df,nome_var):
    r = 0
    nome_variavel = nome_var

    from sklearn.preprocessing import StandardScaler
    sd = StandardScaler()
    #df['PV'] = pd.DataFrame(sd.fit_transform(df['PV'].to_numpy().reshape(1,-1))
    aux = pd.DataFrame(sd.fit_transform(df['PV'].to_numpy().reshape(-1, 1)))
       
    #Calcular desvio padrão    
    std_dev = aux.std()
    mean = aux.mean()    
    
    pv_avg = aux.rolling(window=15).std()

    #Linha do tempo
    fig, ax = plt.subplots()
    #ax.plot(df['PV'])
    ax.plot(pv_avg,color='k')
    xmin,xmax = ax.get_xlim()
    ax.set_ylim(0,6)
    #Linhas de 1 desvio padrão
    min = mean -1*std_dev
    max = mean + 1*std_dev     
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image12.jpg", format='jpg', dpi=100)
    plt.clf()    
    return r

def timelineMoveStd(df,nome_var):
    r = 0
    nome_variavel = nome_var    

    #Calcular desvio padrão    
     
    
    w = 30
    #Desvio padrao médio
    df['MOVE_STD'] = df['PV'].rolling(window=w).std()
    df['MOVE_AVG'] = df['PV'].rolling(window=w).mean()
    df['DIFF_AVG'] = abs(df['MOVE_AVG'].diff() )
    df['DIFF_STD'] = abs(df['MOVE_STD'].diff() )
    df['DIFF_AVG_%'] = abs(df['MOVE_AVG'].diff() /df['MOVE_AVG'])
    df['DIFF_STD_%'] = abs(df['MOVE_STD'].diff() /df['MOVE_STD'])
    
    fator_std = (df['DIFF_STD'].mean() + 3*df['DIFF_STD'].std())
    fator_mean = (df['DIFF_AVG'].mean() + 3*df['DIFF_AVG'].std())  

    df['CHANGE_STD'] = df['DIFF_STD'] / fator_std
    df['CHANGE_AVG'] = df['DIFF_AVG'] / fator_mean

    #Verificar as maiores varia~]oes
    aux = df.sort_values(by=['CHANGE_STD'], ascending=False)
    #aux.to_csv('c:\\ftp\\df.csv',sep=';',decimal=',')
    #print(aux.head(20))
    
     #Linha do tempo
    fig, ax = plt.subplots(3,sharex=True)
    #ax.plot(df['PV'])
    ax[0].plot(df['PV'],color='b')

    #Linhas de segmentação    

    aux = aux[(aux['CHANGE_STD'] > 1) | (aux['CHANGE_AVG'] > 2)]
    for i in range(len(aux)):
        ax[0].axvline(x=aux.iloc[i,0],color='r',linestyle='--')
        #ax[1].axvline(x=aux.iloc[i,0],color='r',linestyle='--')
        #ax[2].axvline(x=aux.iloc[i,0],color='r',linestyle='--')    


    #ax[0].plot(df['MOVE_AVG'],color='k')
    ax[1].plot(df['CHANGE_STD'],color='b')
    ax[2].plot(df['CHANGE_AVG'],color='b')
    #xmin,xmax = ax.get_xlim()

    ax[0].set_ylabel("Dados")
    ax[0].yaxis.set_label_position("right")
    ax[1].set_ylabel("Desvio Padrao")
    ax[1].yaxis.set_label_position("right")
    ax[2].set_ylabel("Média")
    ax[2].yaxis.set_label_position("right")


    plt.savefig(folder+nome_variavel+"_image13.jpg", format='jpg', dpi=100)
    #plt.show()
    plt.clf()    


    
    return r

def timelineOficial(df2,nome_var):
    r = 0
    nome_variavel = nome_var    
    df = df2.copy()
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    
    #Setar coluna datahora como index
    df['DataHora'] = df['datahora']
    df.set_index('DataHora', inplace=True)

    w = 15
    #Desvio padrao médio
    df['MOVE_STD'] = df['PV'].rolling(window=w).std()
    df['MOVE_AVG'] = df['PV'].rolling(window=w).mean()
    df['DIFF_AVG'] = abs(df['MOVE_AVG'].diff() )
    df['DIFF_STD'] = abs(df['MOVE_STD'].diff() )
    df['DIFF_AVG_%'] = abs(df['MOVE_AVG'].diff() /df['MOVE_AVG'])
    df['DIFF_STD_%'] = abs(df['MOVE_STD'].diff() /df['MOVE_STD'])
    
    fator_std = (df['DIFF_STD'].mean() + 2*df['DIFF_STD'].std())
    fator_mean = (df['DIFF_AVG'].mean() + 2*df['DIFF_AVG'].std())  

    df['CHANGE_STD'] = df['DIFF_STD'] / fator_std
    df['CHANGE_AVG'] = df['DIFF_AVG'] / fator_mean
    df['CLUSTER'] = 1

    
    #Verificar as maiores variações
    aux = df.sort_values(by=['CHANGE_STD'], ascending=False)
    #print(aux.head(10))

     #Linha do tempo
    fig, ax = plt.subplots()
    #ax.plot(df['PV'])
    ax.plot(df.index,df['PV'],color='b')    
    ax.plot(df.index,df['MOVE_AVG'],color='k')   
    ax.set_xticklabels(df.index, rotation=45,fontsize=6,ha='right')
    xmin,xmax = ax.get_xlim()
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=24))
    ax.axhline(y=mean,color='g',linestyle='--')
    
    #Linhas de segmentação    
    aux = df[(df['CHANGE_STD'] > 1) | (df['CHANGE_AVG'] > 1)]
    aux = aux.sort_values(by=['datahora'], ascending=True)
    aux['DIFF_TIME'] = aux['datahora'].diff()
    
    #usar um dataframe auxiliar    
    aux = aux[(aux['DIFF_TIME'] >= pd.Timedelta(minutes=15)) | (aux['DIFF_TIME'].isna() == True)]
    #print(aux)
    cluster = 1
    for i in aux.index:
        ax.axvline(x=aux.loc[i,['datahora']],color='r',linestyle='--')
        df.loc[i,'CLUSTER'] = cluster
        cluster = cluster + 1

    # Formatar os rótulos de data para mostrar horas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Rotacionar e alinhar os rótulos de data
    fig.autofmt_xdate()

    # Adicionar espaço extra para os rótulos dos ticks
    plt.subplots_adjust(bottom=0.2)
    plt.text(xmax, mean, 'Mean')
    
    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)
    
    plt.savefig(folder+nome_variavel+"_image1.jpg", format='jpg', dpi=100)
    #plt.show()
    
    plt.clf()    
    r = 1

    #timelineMoveStd(df,nome_var)
    return r

def timelineOficial2(df,nome_var):
    
    
    df.to_csv('c:\\ftp\\df_cluster_2.csv',sep=';',decimal=',')
    r = 0
    nome_variavel = nome_var    
    
    #Calcular média e desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
        
    #Linha do tempo
    fig, ax = plt.subplots()    
    ax.plot(df.index,df['PV'],color='b')        
    ax.plot(df.index,df['MOVE_AVG'],color='k')   
    ax.set_xticklabels(df.index, rotation=45,fontsize=6,ha='right')
    xmin,xmax = ax.get_xlim()    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=24))
    ax.axhline(y=mean,color='g',linestyle='--')

    print("Cluster",len(df['CLUSTER'].unique()))
    
    #Carregar linhas de corte
    for cluster in df['CLUSTER'].unique():        
        momento = df[df['CLUSTER']==cluster].head(1).index
        #print("Cluster",cluster,"Momento",momento)
        ax.axvline(x=momento,color='r',linestyle='--')

    momento = df.tail(1).index
    ax.axvline(x=momento,color='r',linestyle='--')


    import matplotlib.dates as mdates
    # Formatar os rótulos de data para mostrar horas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Rotacionar e alinhar os rótulos de data
    fig.autofmt_xdate()

    # Adicionar espaço extra para os rótulos dos ticks
    plt.subplots_adjust(bottom=0.2)
    plt.text(xmax, mean, 'Mean')

    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)

    plt.savefig(folder+nome_variavel+"_image11.jpg", format='jpg', dpi=100)
    #plt.show()

    plt.clf()    
    r = 1

    #timelineMoveStd(df,nome_var)
    return r

def timelineOficial3(df,nome_var):

    r = 0
    nome_variavel = nome_var    
    
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    
    #Linha do tempo
    fig, ax = plt.subplots()    

    for cluster in df['CLUSTER'].unique():
        cluster_data = df[df['CLUSTER'] == cluster]
        ax.plot(cluster_data.index, cluster_data['PV'], label=f'Cluster {cluster}')

    #ax.plot(df.index,df['PV'],color='b')        
    ax.plot(df.index,df['MOVE_AVG'],color='k')   
    ax.set_xticklabels(df.index, rotation=45,fontsize=6,ha='right')
    xmin,xmax = ax.get_xlim()    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=24))
    ax.axhline(y=mean,color='g',linestyle='--')


    #Carregar linhas de corte
    for cluster in df['CLUSTER'].unique():
        momento = df[df['CLUSTER']==cluster].head(1).index
        ax.axvline(x=momento,color='r',linestyle='--')

    momento = df.tail(1).index
    ax.axvline(x=momento,color='r',linestyle='--')
    
    # Formatar os rótulos de data para mostrar horas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    # Rotacionar e alinhar os rótulos de data
    fig.autofmt_xdate()

    # Adicionar espaço extra para os rótulos dos ticks
    plt.subplots_adjust(bottom=0.2)
    plt.text(xmax, mean, 'Mean')

    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)

    plt.savefig(folder+nome_variavel+"_image2.jpg", format='jpg', dpi=100)
    #plt.show()

    plt.clf()    
    r = 1

    #timelineMoveStd(df,nome_var)
    return r

def boxplot(df,nome_var):
    fig, ax = plt.subplots()  
    

    # Agrupando os dados por 'CLuster'
    groups = df.groupby('CLUSTER')['PV'].apply(list)

    # Criando o boxplot
    ax.boxplot(groups, labels=groups.index)
    ax.set_title("Box Plot")   


    # Calculando a média para cada grupo
    # Calcular média e desvio padrão
    stats = df.groupby('CLUSTER')['PV'].agg(['mean', 'std','count']).T
    stats = stats.round(2)
    # Adicionar tabela com média e desvio padrão
    table_data = stats.values
    row_labels = ['Média', 'Desvio Padrão','Tamanho']
    table = plt.table(cellText=table_data, rowLabels=row_labels, colLabels=stats.columns, 
                      cellLoc='center', loc='bottom', bbox=[0, -0.32, 1, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Ocultando os valores do eixo x
    plt.gca().set_xticklabels([])
    plt.subplots_adjust(left=0.2, bottom=0.25)
    plt.savefig(folder+nome_var+"_image4.jpg", format='jpg', dpi=100)
    plt.clf() 

def boxplot2(df,nome_var):
    fig, ax = plt.subplots()  
    
    df = df.groupby('CLUSTER').apply(treatment.normalizar_clusters).reset_index(drop=True)
    # Agrupando os dados por 'CLuster'
    groups = df.groupby('CLUSTER')['PV'].apply(list)

    # Criando o boxplot
    ax.boxplot(groups, labels=groups.index)
    ax.set_title("Box Plot")   


    # Calculando a média para cada grupo
    # Calcular média e desvio padrão
    stats = df.groupby('CLUSTER')['PV'].agg(['mean', 'std']).T
    stats = stats.round(2)
    # Adicionar tabela com média e desvio padrão
    table_data = stats.values
    row_labels = ['Média', 'Desvio Padrão']
    table = plt.table(cellText=table_data, rowLabels=row_labels, colLabels=stats.columns, 
                      cellLoc='center', loc='bottom', bbox=[0, -0.2, 1, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Ocultando os valores do eixo x
    plt.gca().set_xticklabels([])
    plt.subplots_adjust(left=0.2, bottom=0.25)
    plt.savefig(folder+nome_var+"_image14.jpg", format='jpg', dpi=100)
    plt.clf() 

def boxplot3(df,nome_var):
    fig, ax = plt.subplots()  
    
    df = df.groupby('CLUSTER').apply(treatment.standardize_group).reset_index(drop=True)
    # Agrupando os dados por 'CLuster'
    groups = df.groupby('CLUSTER')['PV'].apply(list)

    # Criando o boxplot
    ax.boxplot(groups, labels=groups.index)
    ax.set_title("Box Plot")   


    # Calculando a média para cada grupo
    # Calcular média e desvio padrão
    stats = df.groupby('CLUSTER')['PV'].agg(['mean', 'std']).T
    stats = stats.round(2)
    # Adicionar tabela com média e desvio padrão
    table_data = stats.values
    row_labels = ['Média', 'Desvio Padrão']
    table = plt.table(cellText=table_data, rowLabels=row_labels, colLabels=stats.columns, 
                      cellLoc='center', loc='bottom', bbox=[0, -0.2, 1, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # Ocultando os valores do eixo x
    plt.gca().set_xticklabels([])
    plt.subplots_adjust(left=0.2, bottom=0.25)
    plt.savefig(folder+nome_var+"_image15.jpg", format='jpg', dpi=100)
    plt.clf() 

def desvPadMovel(df,nome_var):
    r = 0
    nome_variavel = nome_var

    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()    
    
    #Linha do tempo
    fig, axs = plt.subplots(3,1,sharex=True)
    #ax.plot(df['PV'])
    axs[0].plot(df['PV'])
    axs[1].plot(df['DIFF_STD_%'])
    axs[2].plot(df['DIFF_AVG_%'])

    axs[0].set_title('PV')
    axs[1].set_title('DIFF_STD_%')
    axs[2].set_title('DIFF_AVG_%')
    plt.tight_layout()

    # Formatar os rótulos de data para mostrar horas
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    axs[2].xaxis.set_major_locator
    # Rotacionar e alinhar os rótulos de data
    fig.autofmt_xdate()

    # Adicionar espaço extra para os rótulos dos ticks
    plt.subplots_adjust(bottom=0.2)

    
    plt.savefig(folder+nome_variavel+"_image100.jpg", format='jpg', dpi=100)
    plt.clf()    
    return r

def heatmap(df,nome):
     #TEMPORARIO -> criar HEATMAP
    fig, ax = plt.subplots()  
    # Agrupando os dados por 'CLuster'
    stats = df.groupby('CLUSTER')['PV'].agg(['mean', 'std'])
    stats = stats.round(2)
    print(stats)
    im = ax.imshow(stats.to_numpy())
    #ax.set_xticks(np.arange(len(stats['mean'].columns)))
    #ax.set_yticks(np.arange(len(stats['mean'].columns)))
    # Create colorbar
    cbarlabel = nome
    cbar_kw =  {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")


    fig.tight_layout()
    
    #plt.show()
    plt.savefig(folder+nome+"_image40.jpg", format='jpg', dpi=100)

def fatoresAVG(df,nome_var):
    fig, axs = plt.subplots(4,1)  

    axs[0].plot(df['PV'])
    axs[1].plot(df['MOVE_AVG'])
    axs[2].plot(df['DIFF_AVG'])
    axs[3].plot(df['CHANGE_AVG'])

    axs[0].set_title('PV')
    axs[1].set_title('MOVE_AVG')
    axs[2].set_title('DIFF_AVG')
    axs[3].set_title('CHANGE_AVG')

    axs[3].axhline(y=1,color='r',linestyle='--')

       # Adicionar asteriscos vermelhos nos pontos acima de 1
    y = df['CHANGE_AVG']
    x = df.index
    count = 0
    for i in range(len(y)):
        if y[i] > 1:
            axs[3].plot(x[i], y[i], 'r*')
            count+=1

    # Adicionar anotação no campo superior direito
    plt.annotate(f'* Rupturas: {count}', xy=(0.95, 0.95), xycoords='axes fraction',
             horizontalalignment='right', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(folder+nome_var+"_image8.jpg", format='jpg', dpi=100)

def fatoresSTD(df,nome_var):
    fig, axs = plt.subplots(4,1)  

    axs[0].plot(df['PV'])
    axs[1].plot(df['MOVE_STD'])
    axs[2].plot(df['DIFF_STD'])
    axs[3].plot(df['CHANGE_STD'])

    axs[0].set_title('PV')
    axs[1].set_title('MOVE_STD')
    axs[2].set_title('DIFF_STD')
    axs[3].set_title('CHANGE_STD')

    axs[3].axhline(y=1,color='r',linestyle='--')

    # Adicionar asteriscos vermelhos nos pontos acima de 1
    y = df['CHANGE_STD']
    x = df.index
    count = 0
    for i in range(len(y)):
        if y[i] > 1:
            axs[3].plot(x[i], y[i], 'r*')
            count+=1

    # Adicionar anotação no campo superior direito
    plt.annotate(f'* Rupturas: {count}', xy=(0.95, 0.95), xycoords='axes fraction',
             horizontalalignment='right', verticalalignment='top')

    plt.tight_layout()
    plt.savefig(folder+nome_var+"_image9.jpg", format='jpg', dpi=100)

def heatmap2(df,nome_var,list_cluster):
    #Criar tabela de resultados
    clusters = np.array(list_cluster)
    result = pd.DataFrame(index=clusters, columns=clusters,dtype=float)
    #Execuar analise para cada segmento
    for i in range(len(list_cluster)):
        for j in range(len(list_cluster)):
            cluster_A = list_cluster[i]
            cluster_B = list_cluster[j]
            A = df[df['CLUSTER']==cluster_A]['PV'].mean()
            B = df[df['CLUSTER']==cluster_B]['PV'].mean()        
            p = round(A/B,1)
            result.iloc[i,j] = p
    
    fig, ax = plt.subplots()
    im = ax.imshow(result.to_numpy())
    ax.set_xticks(np.arange(len(result.columns)))
    ax.set_yticks(np.arange(len(result.columns)))
    
    # Create colorbar
    cbarlabel = 'p_value'
    cbar_kw =  {}
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    fig.tight_layout()    
    #plt.show()
    plt.savefig(folder+nome_var+"_image4.jpg", format='jpg', dpi=100)    
    r = 1
    return r


def Fourier(df,nome_var):
    df = treatment.remove_outlier(df,"PV")
    s = df['PV']
    
    N = len(s) #Tamanho dos dados
    dt = 1
    t= np.arange(0,N,dt)
    yf=np.fft.fftfreq(N,dt)  # a nice helper function to get the frequencies  
    xf=np.fft.fft(s)

    #ifft
    plt.figure(figsize=(12,8))
    plt.subplot(512)
    plt.plot(t,s)
    plt.title('Sinal Original')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')

    plt.subplot(514)
    plt.plot(yf ,xf)
    plt.title('Espectro de Frequência')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0,0.01)
    plt.savefig(folder+nome_var+"_Fourier.jpg", format='jpg', dpi=100)