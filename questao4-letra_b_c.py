
### Bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


### Abrindo os arquivos ###

arquivo = open('dados/dataX.dat', 'r')
dados = []
for linha in arquivo:
    colunas = linha.split()
    dados.append(float(colunas[0]))
    
n=len(dados)
sigma=np.arange(0,10,0.1)
l1=len(sigma)
sigma2=np.zeros(l1)
for i in range(l1):
    sigma2[i]=sigma[i]**2.0
mu=np.arange(-10,10,0.1)

l2=len(mu)
lnL=np.zeros((l1,l2))

for i in range(l1):
    for j in range(l2):
        lnL[i,j]=n*np.log(1.0/(sigma[i]*(2.0*np.pi)**0.5))-(1.0/(2.0*sigma[i]**2.0))*(sum((dados-mu[j])**2.0))
                                                                                  


fig=plt.figure(figsize=(10, 10))
ax=fig.add_subplot(111, projection='3d')
for i in range(l1):
    for j in range(l2):
        ax.scatter3D(sigma2[i],mu[j],lnL[i,j],c='purple',cmap='Spectral')

ax.set_title('Log-verossimilhança', fontsize=18)
ax.set_xlabel('sigma2', fontsize=15)
ax.set_ylabel('mu', fontsize=15)
ax.set_zlabel('ln(L) - log-verossimilhança', fontsize=15)# [view_init] Modifica o ângulo de visualização do gráfico
ax.view_init(-20,35)
plt.show()                                                                                   

### estimativas de mu e sigma ###

Xbarra=0.0
S2=0.0

for i in range(n):
    Xbarra=Xbarra+dados[i]/n
for i in range(n):
    S2=S2+((dados[i]-Xbarra)**2.0)/(n-1)

print("Média estimada = ",Xbarra)
print("Variância estimada = ",S2)
                                                                                     
