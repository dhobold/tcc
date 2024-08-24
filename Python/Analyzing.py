import ProgressBar as pb
import Statistics as statistics

def Do(df_var,big_df):
    #Quantas variaveis serão analisadas
    tam = len(df_var['variavel'])
    
    #Progress Bar    
    print("Iniciando analise dos dados.")    
    pb.printProgressBar(0, tam, prefix = 'Progress:', suffix = 'Complete', length = 50)
    

    #Gerar gráficos e Score Para cada variavel existente no DataFrame
    for i in range(tam):    
        nomevar = df_var['variavel'].iloc[i]
        aux_df = big_df[big_df['variavel']==nomevar]
           
        #Gerar estatistica + gráficos da variavel
        title,count,mean,mode,stdDev,r0,r1,r2,r3,r4 = statistics.Do(aux_df,nomevar)
        
        #Carregar dados basicos    
        df_var.loc[df_var['variavel']==nomevar,'count'] = count
        df_var.loc[df_var['variavel']==nomevar,'mean'] = mean
        df_var.loc[df_var['variavel']==nomevar,'mode']= mode
        df_var.loc[df_var['variavel']==nomevar,'stdDev'] = stdDev
        #Carregar score de cada analises
        df_var.loc[df_var['variavel']==nomevar,'r0'] = r0
        df_var.loc[df_var['variavel']==nomevar,'r1'] = r1
        df_var.loc[df_var['variavel']==nomevar,'r2'] = r2
        df_var.loc[df_var['variavel']==nomevar,'r3'] = r3
        df_var.loc[df_var['variavel']==nomevar,'r4'] = r4
        #Calcula score média
        df_var.loc[df_var['variavel']==nomevar,'Score'] = (r0+r1+r2+r3+r4)/5
        df_var.loc[df_var['variavel']==nomevar,'done'] = 1
        
        #Atualizar ProgressBar
        pb.printProgressBar(i+1, tam, prefix = 'Progress:', suffix = 'Complete', length = 50)
    print("Fim da analise dos dados.")
    #print(df_var)
    #Ordernar Variaveis conforme SCORE de anormalidade
    df_var = df_var[df_var['done']==1]
    df_var.sort_values(by='Score',ascending=False,inplace=True)
    statistics.doTable(df_var[['variavel','count','mean','mode','r0','r1','r2','r3','r4','Score']])
   
    #Retornar DataFrame analisado
    return df_var

