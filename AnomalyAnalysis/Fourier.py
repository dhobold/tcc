# fourier filter example (1D)

import matplotlib.pyplot as p
import numpy as np
import Ingestion as ingestion
import Treatment as treatment

big_df = ingestion.Do()
big_df,df_var = treatment.Do(big_df)

#Especificar quais variaveis queremos gerar o relatorio
big_df = big_df[big_df['nivel_1']=='Linha 1']
big_df = big_df[big_df['nivel_2']=='COS']
big_df = big_df[big_df['nivel_3']=='Ciclo']
big_df = big_df[big_df['nivel_4']=='Maquina']

print(big_df.head(10))

s = big_df['PV']
dt = 1
t= np.arange(0,5,dt)
f1,f2= 5, 20  #Hz
n = len(big_df['datahora'])
#fft
s-= s.mean()  # remove DC (spectrum easier to look at)
fr=np.fft.fftfreq(n,dt)  # a nice helper function to get the frequencies  
fou=np.fft.fft(s)
#make up a narrow bandpass with a Gaussian
df=0.1
gpl= np.exp(- ((fr-f1)/(2*df))**2)+ np.exp(- ((fr-f2)/(2*df))**2)  # pos. frequencies
gmn= np.exp(- ((fr+f1)/(2*df))**2)+ np.exp(- ((fr+f2)/(2*df))**2)  # neg. frequencies
g=gpl+gmn    
filt=fou*g  #filtered spectrum = spectrum * bandpass 

#ifft
s2=np.fft.ifft(filt)

p.figure(figsize=(12,8))

p.subplot(512)
p.plot(t,s)
p.title('data w/ noise')

p.subplot(513)
p.plot(np.fft.fftshift(fr) ,np.fft.fftshift(np.abs(fou) )  )
p.title('spectrum of noisy data')

p.subplot(514)
p.plot(fr,g*50, 'r')  
p.plot(fr,np.abs(filt))
p.title('filter (red)  + filtered spectrum')

p.subplot(515)
p.plot(t,np.real(s2))
p.title('filtered time data')

p.show()