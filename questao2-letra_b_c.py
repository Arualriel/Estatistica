
### Bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import pandas as pd 

### Abrindo os arquivos ###

arquivoq = open('dados/data1q.dat', 'r')
dadosq = []
for linha in arquivoq:
    colunas = linha.split()
    dadosq.append(float(colunas[0]))

arquivox = open('dados/data1x.dat', 'r')
dadosx = []
for linha in arquivox:
    colunas = linha.split()
    dadosx.append(float(colunas[0]))

arquivoy = open('dados/data1y.dat', 'r')
dadosy = []
for linha in arquivoy:
    colunas = linha.split()
    dadosy.append(float(colunas[0]))

arquivot = open('dados/data1t.dat', 'r')
dadost = []
for linha in arquivot:
    colunas = linha.split()
    dadost.append(float(colunas[0]))


### Estimativas das médias ###

Qbarra=0.0
Xbarra=0.0
Ybarra=0.0
Tbarra=0.0

nq=len(dadosq)
nx=len(dadosx)
ny=len(dadosy)
nt=len(dadost)

for i in range(nq):
    Qbarra=Qbarra+dadosq[i]/nq

for i in range(nx):
    Xbarra=Xbarra+dadosx[i]/nx

for i in range(ny):
    Ybarra=Ybarra+dadosy[i]/ny

for i in range(nt):
    Tbarra=Tbarra+dadost[i]/nt


### Estimativas das variâncias ###

S2q=0.0
S2x=0.0
S2y=0.0
S2t=0.0

for i in range(nq):
    S2q=S2q+((dadosq[i]-Qbarra)**2.0)/(nq-1)

for i in range(nx):
    S2x=S2x+((dadosx[i]-Xbarra)**2.0)/(nx-1)

for i in range(ny):
    S2y=S2y+((dadosy[i]-Ybarra)**2.0)/(ny-1)

for i in range(nt):
    S2t=S2t+((dadost[i]-Tbarra)**2.0)/(nt-1)


### Estimativas dos mínimos e dos máximos ###

Kq=min(dadosq)
Kx=min(dadosx)
Ky=min(dadosy)
Kt=min(dadost)

Mq=max(dadosq)
Mx=max(dadosx)
My=max(dadosy)
Mt=max(dadost)

### Gráficos - histogramas + pdfs ###

muq=1.0
mux=0.0
muy=8.0
mut=2.5

sigma2q=2.0
sigma2x=1.0/3.0
sigma2y=64.0
sigma2t=1.875

print(Mq,Kq,Mx,Kx,My,Ky,Mt,Kt)

pdfq=np.zeros(nq)
q=np.arange(Kq,Mq,(Mq-Kq)/nq)
mu=muq
sigma2=sigma2q

pdfx=np.zeros(nx)
x=np.arange(Kx,Mx,(Mx-Kx)/nx)
a=-1.0
b=1.0

pdfy=np.zeros(ny)
y=np.arange(Ky,My,(My-Ky)/ny)
gamma=1.0/8.0

N=10.0
p=0.25
pdft=[]
npdft=[]
kpdft=[]


for i in range(nq):
    pdfq[i]=(1.0/((sigma2**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((q[i]-mu)/(sigma2**0.5))**2.0)
    
for i in range(nx):
    if((x[i]<a) or (x[i]>b)):
        pdfx[i]=0.0
    else:
        pdfx[i]=1.0/(b-a)
    
for i in range(ny):
    if(y[i]<=0):
        pdfy[i]=0.0
    else:
        pdfy[i]=gamma*np.exp(-gamma*y[i])
        
i=0

while(i<nt):
    kpdft.append(dadost[i])
    npdft.append(factorial(N)/(factorial(dadost[i])*factorial(N-dadost[i])))
    pdft.append(npdft[i]*(p**dadost[i])*((1-p)**(N-dadost[i])))
    i=i+1

o=0.0

somafaq=0
faq = pd.Series(dadosq).value_counts()
for i in faq:
    somafaq=somafaq+i

somafax=0
fax = pd.Series(dadosx).value_counts()
for i in fax:
    somafax=somafax+i

somafay=0
fay = pd.Series(dadosy).value_counts()
for i in fay:
    somafay=somafay+i

somafat=0
fat = pd.Series(dadost).value_counts()
for i in fat:
    somafat=somafat+i


plt.title('Dados Q')
plt.xlabel('q')
plt.ylabel('Frequência relativa')
plt.hist(dadosq, weights=np.zeros_like(dadosq)+1.0/somafaq, bins= 50, rwidth=0.5, color='red', label='Histograma de Q')
plt.show()

plt.title('Dados Q')
plt.xlabel('q')
plt.ylabel('Probabilidade')
plt.plot(q,pdfq, color='red', label='PDF de Q')
plt.scatter(Qbarra,o,color='black',label='Média estimada')
plt.scatter(muq,o,color='orange',label='Média real')
plt.show()

plt.title('Dados X')
plt.xlabel('x')
plt.ylabel('Frequência relativa')
plt.hist(dadosx, weights=np.zeros_like(dadosx)+1.0/somafax,bins= 50, rwidth=0.5, color='blue', label='Histograma de X')
plt.show()

plt.title('Dados X')
plt.xlabel('x')
plt.ylabel('Probabilidade')
plt.plot(x,pdfx, color='blue', label='PDF de X')
plt.scatter(Xbarra,o,color='black',label='Média estimada')
plt.scatter(mux,o,color='orange',label='Média real')
plt.show()

plt.title('Dados Y')
plt.xlabel('y')
plt.ylabel('Frequência relativa')
plt.hist(dadosy, weights=np.zeros_like(dadosy)+1.0/somafay,bins= 50, rwidth=0.5, color='green', label='Histograma de Y')
plt.show()

plt.title('Dados Y')
plt.xlabel('y')
plt.ylabel('Probabilidade')
plt.plot(y,pdfy, color='green', label='PDF de Y')
plt.scatter(Ybarra,o,color='black',label='Média estimada')
plt.scatter(muy,o,color='orange',label='Média real')
plt.show()

plt.title('Dados T')
plt.xlabel('t')
plt.ylabel('Frequência relativa')
plt.hist(dadost, weights=np.zeros_like(dadost)+1.0/somafat,bins= 50, rwidth=0.5, color='purple', label='Histograma de T')
plt.show()

plt.title('Dados T')
plt.xlabel('t')
plt.ylabel('Probabilidade')
plt.plot(dadost,pdft, color='purple', label='PDF de T')
plt.scatter(Tbarra,o,color='black',label='Média estimada')
plt.scatter(mut,o,color='orange',label='Média real')
plt.show()

print("Qbarra=",Qbarra,"muq=",muq,"S2q=",S2q,"sigma2q=",sigma2q)

print("Xbarra=",Xbarra,"mux=",mux,"S2x=",S2x,"sigma2x=",sigma2x)

print("Ybarra=",Ybarra,"muy=",muy,"S2y=",S2y,"sigma2y=",sigma2y)

print("Tbarra=",Tbarra,"mut=",mut,"S2t=",S2t,"sigma2t=",sigma2t)
