
from matplotlib import pyplot as plt, table
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import Treatment as treatment

#folder = "c:\\ftp\\Reports\\Images\\"
folder =  "C:\\Users\\pohed\\OneDrive\\DIH\\Pós FACENS\\TCC\\Reports\\Images\\"
######################################################
#Gráfico 1 - Linha do tempo mais 3x desvio padrão
######################################################

def timeLine(df,nome_var):
    nome_variavel = nome_var
    #Calcular desvio padrão    
    std_dev = df['PV'].std()
    mean = df['PV'].mean()
    
    min = mean -3*std_dev
    max = mean + 3*std_dev    
    #Linha do tempo
    fig, ax = plt.subplots()
    ax.plot(df['PV'])
    ax.axhline(y=min,color='r',linestyle='--')
    ax.axhline(y=max,color='r',linestyle='--')
    ax.axhline(y=mean,color='r',linestyle='--')
    ax.set_ylim(mean - 6*std_dev,mean + 6*std_dev)

    xmin,xmax = ax.get_xlim()
    
    plt.text(xmax, min, '-3dev')
    plt.text(xmax, max, '+3dev')
    plt.text(xmax, mean, 'Mean')    
    #plt.text(xmax, mean*0.95, ''+str(round(mean,2)))    
    
    ax.set_title("Linha do tempo")    
    plt.savefig(folder+nome_variavel+"_image1.jpg", format='jpg', dpi=100)
    plt.clf()
    #Calcular quantos pontos estão dentro ou fora do desvio padrao
    df['r1'] = np.where(df['PV']>max,'NOK','OK')
    df['r1'] = np.where(df['PV']<min,'NOK','OK')
    r = len(df[df['r1']=='NOK']) / len(df['variavel'])
    return r


def desvioPadraoMovel(df,nome_var):
    #Tratar dados de ciclo
    df = treatment.cycle(df)
    
    nome_variavel = nome_var
    #Calcula desvio padrão do dia    
    df_std = df.groupby(['data','hora','variavel'])['PV'].std().reset_index()

    #|Analise do desvio padrao
    fig, ax = plt.subplots()
    ax.set_title("Desvio Padrao")
    ax.plot(df_std['PV'])
    z = np.polyfit(df_std.index,df_std['PV'],1)
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
    
    
    nome_variavel = nome_var
    #Analise de anormalidade
    df = df.groupby(['data','hora','variavel'])['PV'].mean().reset_index()
    #df.groupby(['data','hora','variavel'])['PV'].agg(pd.Series.mode).to_frame()
    
    outliers_fraction = float(.01)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df['PV'].to_numpy().reshape(-1,1))
    data = pd.DataFrame(np_scaled)
    #print(data)

    # train isolation forest
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data) 

    df['anomaly'] = model.predict(data)

    # visualization
    #figsize=(10,6)
    fig, ax = plt.subplots()

    a = df.loc[df['anomaly'] == -1, ['PV']] #anomaly

    ax.plot(df.index, df['PV'], color='black', label = 'Normal')
    ax.scatter(a.index,a['PV'], color='red', label = 'Anomaly')
    plt.legend()
    plt.title("Detecção de anormalidade no desvio padrao")
    plt.savefig(folder+nome_variavel+"_image4.jpg", format='jpg', dpi=100)
    #plt.show();
    plt.clf()
    
    r = len(a)/len(df)
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
