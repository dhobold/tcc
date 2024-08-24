from matplotlib import pyplot as plt, table
import pandas as pd
import Treatment as tr
import Charts as ch


#Definir local que serão salvo as imagens
#folder = "c:\\ftp\\Reports\\Images\\"
folder  = "C:\\Users\\pohed\\OneDrive\\DIH\\Pós FACENS\\TCC\\Reports\\"




def Do(aux_df,nome_var):     
    df = aux_df
    df = tr.remove_outlier(aux_df,'PV')
    nome_variavel = nome_var
    
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
    
    #print(nome_variavel)
    #print(df['PV'].count())
    ######################################################
    #Gráfico 1 - TimeLine
    ######################################################
    r1 = ch.timeLine(df,nome_var)    

    ######################################################
    #Gráfico 2 - Desvio padrão ao longo do tempo
    ######################################################
    r2 = ch.desvioPadraoMovel(df,nome_var)
        
    ######################################################
    #Gráfico 3 - Histrograma    
    ######################################################
    r3 = ch.histograma(df,nome_var)
    
    ######################################################
    #Gráfico 4 - Analise de anormalidade
    ######################################################
    r4 = ch.anormalidade(df,nome_var)
    
    df['PV'].to_csv('c:\\ftp\\PV.csv',sep=';',decimal=',')
    count = df['PV'].count()
    mean = df['PV'].mean()
    mode = df['PV'].mode()[0]
    
    stdDev = df['PV'].std()
    r0 = mean/mode
    
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