
### Bibliotecas ###

import numpy as np
import matplotlib.pyplot as plt
import random as rd
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

n5=5
n10=10
n50=50

mu=1.0
sigma2=2.0

a=-1.0
b=1.0

gamma=1.0/8.0

N=10.0
p=0.25

M=10000

amostrasq5=np.zeros((n5,M))
amostrasq10=np.zeros((n10,M))
amostrasq50=np.zeros((n50,M))

amostrasx5=np.zeros((n5,M))
amostrasx10=np.zeros((n10,M))
amostrasx50=np.zeros((n50,M))

amostrasy5=np.zeros((n5,M))
amostrasy10=np.zeros((n10,M))
amostrasy50=np.zeros((n50,M))

amostrast5=np.zeros((n5,M))
amostrast10=np.zeros((n10,M))
amostrast50=np.zeros((n50,M))

rng = np.random.default_rng()
for j in range(M):
    for i in range(n5):
        amostrasq5[i,j]=rd.normalvariate(mu,sigma2)
        amostrasx5[i,j]=rd.uniform(a,b)
        amostrasy5[i,j]=rd.expovariate(gamma)
    amostrast5[:,j]=rng.binomial(N,p,n5)
    for i in range(n10):
        amostrasq10[i,j]=rd.normalvariate(mu,sigma2)
        amostrasx10[i,j]=rd.uniform(a,b)
        amostrasy10[i,j]=rd.expovariate(gamma)
    amostrast10[:,j]=rng.binomial(N,p,n10)
    for i in range(n50):
        amostrasq50[i,j]=rd.normalvariate(mu,sigma2)
        amostrasx50[i,j]=rd.uniform(a,b)
        amostrasy50[i,j]=rd.expovariate(gamma)
    amostrast50[:,j]=rng.binomial(N,p,n50)



### Estimativas das médias ###


qbarra5=np.zeros(M)
xbarra5=np.zeros(M)
ybarra5=np.zeros(M)
tbarra5=np.zeros(M)

qbarra10=np.zeros(M)
xbarra10=np.zeros(M)
ybarra10=np.zeros(M)
tbarra10=np.zeros(M)

qbarra50=np.zeros(M)
xbarra50=np.zeros(M)
ybarra50=np.zeros(M)
tbarra50=np.zeros(M)


for j in range(M):
    for i in range(n5): 
        qbarra5[j]=qbarra5[j]+amostrasq5[i,j]/n5
        xbarra5[j]=xbarra5[j]+amostrasx5[i,j]/n5
        ybarra5[j]=ybarra5[j]+amostrasy5[i,j]/n5
        tbarra5[j]=tbarra5[j]+amostrast5[i,j]/n5
        
        
    for i in range(n10):
        qbarra10[j]=qbarra10[j]+amostrasq10[i,j]/n10
        xbarra10[j]=xbarra10[j]+amostrasx10[i,j]/n10
        ybarra10[j]=ybarra10[j]+amostrasy10[i,j]/n10
        tbarra10[j]=tbarra10[j]+amostrast10[i,j]/n10
        

    for i in range(n50):
        qbarra50[j]=qbarra50[j]+amostrasq50[i,j]/n50
        xbarra50[j]=xbarra50[j]+amostrasx50[i,j]/n50
        ybarra50[j]=ybarra50[j]+amostrasy50[i,j]/n50
        tbarra50[j]=tbarra50[j]+amostrast50[i,j]/n50
        
        
somafaq5=0
faqs25 = pd.Series(qbarra5).value_counts()
for i in faqs25:
    somafaq5=somafaq5+i

somafax5=0
faxs25 = pd.Series(xbarra5).value_counts()
for i in faxs25:
    somafax5=somafax5+i

somafay5=0
fays25 = pd.Series(ybarra5).value_counts()
for i in fays25:
    somafay5=somafay5+i

somafat5=0
fats25 = pd.Series(tbarra5).value_counts()
for i in fats25:
    somafat5=somafat5+i
    
    
somafaq10=0
faqs210 = pd.Series(qbarra10).value_counts()
for i in faqs25:
    somafaq10=somafaq10+i

somafax10=0
faxs210 = pd.Series(xbarra10).value_counts()
for i in faxs210:
    somafax10=somafax10+i

somafay10=0
fays210 = pd.Series(ybarra10).value_counts()
for i in fays210:
    somafay10=somafay10+i

somafat10=0
fats210 = pd.Series(tbarra10).value_counts()
for i in fats210:
    somafat10=somafat10+i
    
    
somafaq50=0
faqs250 = pd.Series(qbarra50).value_counts()
for i in faqs250:
    somafaq50=somafaq50+i

somafax50=0
faxs250 = pd.Series(xbarra50).value_counts()
for i in faxs250:
    somafax50=somafax50+i

somafay50=0
fays250 = pd.Series(ybarra50).value_counts()
for i in fays250:
    somafay50=somafay50+i

somafat50=0
fats250 = pd.Series(tbarra50).value_counts()
for i in fats250:
    somafat50=somafat50+i

### Estimativas das variâncias ###

s2q5=np.zeros(M)
s2x5=np.zeros(M)
s2y5=np.zeros(M)
s2t5=np.zeros(M)

s2q10=np.zeros(M)
s2x10=np.zeros(M)
s2y10=np.zeros(M)
s2t10=np.zeros(M)

s2q50=np.zeros(M)
s2x50=np.zeros(M)
s2y50=np.zeros(M)
s2t50=np.zeros(M)


for j in range(M):

    for i in range(n5):
        s2q5[j]=s2q5[j]+((amostrasq5[i,j]-qbarra5[j])**2.0)/(n5-1)
        s2x5[j]=s2x5[j]+((amostrasx5[i,j]-xbarra5[j])**2.0)/(n5-1)
        s2y5[j]=s2y5[j]+((amostrasy5[i,j]-ybarra5[j])**2.0)/(n5-1)
        s2t5[j]=s2t5[j]+((amostrast5[i,j]-tbarra5[j])**2.0)/(n5-1)
    

    for i in range(n10):
        s2q10[j]=s2q10[j]+((amostrasq10[i,j]-qbarra10[j])**2.0)/(n10-1)
        s2x10[j]=s2x10[j]+((amostrasx10[i,j]-xbarra10[j])**2.0)/(n10-1)
        s2y10[j]=s2y10[j]+((amostrasy10[i,j]-ybarra10[j])**2.0)/(n10-1)
        s2t10[j]=s2t10[j]+((amostrast10[i,j]-tbarra10[j])**2.0)/(n10-1)
    
    
    
    for i in range(n50):
        s2q50[j]=s2q50[j]+((amostrasq50[i,j]-qbarra50[j])**2.0)/(n50-1)
        s2x50[j]=s2x50[j]+((amostrasx50[i,j]-xbarra50[j])**2.0)/(n50-1)
        s2y50[j]=s2y50[j]+((amostrasy50[i,j]-ybarra50[j])**2.0)/(n50-1)
        s2t50[j]=s2t50[j]+((amostrast50[i,j]-tbarra50[j])**2.0)/(n50-1)

somafaqs25=0
faqs25 = pd.Series(s2q5).value_counts()
for i in faqs25:
    somafaqs25=somafaqs25+i

somafaxs25=0
faxs25 = pd.Series(s2x5).value_counts()
for i in faxs25:
    somafaxs25=somafaxs25+i

somafays25=0
fays25 = pd.Series(s2y5).value_counts()
for i in fays25:
    somafays25=somafays25+i

somafats25=0
fats25 = pd.Series(s2t5).value_counts()
for i in fats25:
    somafats25=somafats25+i
    
    
somafaqs210=0
faqs210 = pd.Series(s2q10).value_counts()
for i in faqs25:
    somafaqs210=somafaqs210+i

somafaxs210=0
faxs210 = pd.Series(s2x10).value_counts()
for i in faxs210:
    somafaxs210=somafaxs210+i

somafays210=0
fays210 = pd.Series(s2y10).value_counts()
for i in fays210:
    somafays210=somafays210+i

somafats210=0
fats210 = pd.Series(s2t10).value_counts()
for i in fats210:
    somafats210=somafats210+i
    
    
somafaqs250=0
faqs250 = pd.Series(s2q50).value_counts()
for i in faqs250:
    somafaqs250=somafaqs250+i

somafaxs250=0
faxs250 = pd.Series(s2x50).value_counts()
for i in faxs250:
    somafaxs250=somafaxs250+i

somafays250=0
fays250 = pd.Series(s2y50).value_counts()
for i in fays250:
    somafays250=somafays250+i

somafats250=0
fats250 = pd.Series(s2t50).value_counts()
for i in fats250:
    somafats250=somafats250+i

### pdfs esperadas (medias) ###

Kq5=min(qbarra5)-5
Kx5=min(xbarra5)-5
Ky5=min(ybarra5)-5
Kt5=min(tbarra5)-5

Mq5=max(qbarra5)+5
Mx5=max(xbarra5)+5
My5=max(ybarra5)+5
Mt5=max(tbarra5)+5

Kq10=min(qbarra10)-5
Kx10=min(xbarra10)-5
Ky10=min(ybarra10)-5
Kt10=min(tbarra10)-5

Mq10=max(qbarra10)+5
Mx10=max(xbarra10)+5
My10=max(ybarra10)+5
Mt10=max(tbarra10)+5

Kq50=min(qbarra50)-5
Kx50=min(xbarra50)-5
Ky50=min(ybarra50)-5
Kt50=min(tbarra50)-5

Mq50=max(qbarra50)+5
Mx50=max(xbarra50)+5
My50=max(ybarra50)+5
Mt50=max(tbarra50)+5

q5=np.arange(Kq5,Mq5,(Mq5-Kq5)/M)
q10=np.arange(Kq10,Mq10,(Mq10-Kq10)/M)
q50=np.arange(Kq50,Mq50,(Mq50-Kq50)/M)

x5=np.arange(Kx5,Mx5,(Mx5-Kx5)/M)
x10=np.arange(Kx10,Mx10,(Mx10-Kx10)/M)
x50=np.arange(Kx50,Mx50,(Mx50-Kx50)/M)

y5=np.arange(Ky5,My5,(My5-Ky5)/M)
y10=np.arange(Ky10,My10,(My10-Ky10)/M)
y50=np.arange(Ky50,My50,(My50-Ky50)/M)

t5=np.arange(Kt5,Mt5,(Mt5-Kt5)/M)
t10=np.arange(Kt10,Mt10,(Mt10-Kt10)/M)
t50=np.arange(Kt50,Mt50,(Mt50-Kt50)/M)


pdfq5=np.zeros(M)

pdfq10=np.zeros(M)

pdfq50=np.zeros(M)

for i in range(M):
    pdfq5[i]=(1.0/((sigma2**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((q5[i]-mu)/(sigma2**0.5))**2.0)

for i in range(M):
    pdfq10[i]=(1.0/((sigma2**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((q10[i]-mu)/(sigma2**0.5))**2.0)

for i in range(M):
    pdfq50[i]=(1.0/((sigma2**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((q50[i]-mu)/(sigma2**0.5))**2.0)



pdfx5=np.zeros(M)

pdfx10=np.zeros(M)

pdfx50=np.zeros(M)

Ex=(a+b)/2
sx=((b-a)**2.0)/12.0

for i in range(M):
    pdfx5[i]=(1.0/((sx**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((x5[i]-Ex)/(sx**0.5))**2.0)

for i in range(M):
    pdfx10[i]=(1.0/((sx**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((x10[i]-Ex)/(sx**0.5))**2.0)

for i in range(M):
    pdfx50[i]=(1.0/((sx**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((x50[i]-Ex)/(sx**0.5))**2.0)



pdfy5=np.zeros(M)

pdfy10=np.zeros(M)

pdfy50=np.zeros(M)

Ey=1.0/gamma
sy=1.0/(gamma**2.0)

for i in range(M):
    pdfy5[i]=(1.0/((sy**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((y5[i]-Ey)/(sy**0.5))**2.0)

for i in range(M):
    pdfy10[i]=(1.0/((sy**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((y10[i]-Ey)/(sy**0.5))**2.0)

for i in range(M):
    pdfy50[i]=(1.0/((sy**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((y50[i]-Ey)/(sy**0.5))**2.0)


pdft5=np.zeros(M)

pdft10=np.zeros(M)

pdft50=np.zeros(M)

Et=N*p
st=N*p*(1-p)

for i in range(M):
    pdft5[i]=(1.0/((st**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((t5[i]-Et)/(st**0.5))**2.0)

for i in range(M):
    pdft10[i]=(1.0/((st**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((t10[i]-Et)/(st**0.5))**2.0)

for i in range(M):
    pdft50[i]=(1.0/((st**0.5)*((2*np.pi)**0.5)))*np.exp((-1.0/2.0)*((t50[i]-Et)/(st**0.5))**2.0)



### graficos ###

## medias ##


plt.title('Amostras Q n=5')
plt.xlabel('qbarra n=5')
plt.ylabel('Frequência relativa')
plt.hist(qbarra5, weights=np.zeros_like(qbarra5)+1.0/somafaq5,bins= 100, rwidth=0.5, color='red', label='Histograma de qbarra')
plt.show()
plt.title('Amostras Q n=5')
plt.xlabel('qbarra n=5')
plt.ylabel('PDF de qbarra')
plt.plot(q5,pdfq5,color='red')
plt.show()

plt.title('Amostras X n=5')
plt.xlabel('xbarra n=5')
plt.ylabel('Frequência relativa')
plt.hist(xbarra5, weights=np.zeros_like(xbarra5)+1.0/somafax5,bins= 100, rwidth=0.5, color='blue', label='Histograma de xbarra')
plt.show()
plt.title('Amostras X n=5')
plt.xlabel('xbarra n=5')
plt.ylabel('PDF de xbarra')
plt.plot(x5,pdfx5,color='blue')
plt.show()

plt.title('Amostras Y n=5')
plt.xlabel('ybarra n=5')
plt.ylabel('Frequência relativa')
plt.hist(ybarra5, weights=np.zeros_like(ybarra5)+1.0/somafay5,bins= 100, rwidth=0.5, color='green', label='Histograma de ybarra')
plt.show()
plt.title('Amostras Y n=5')
plt.xlabel('ybarra n=5')
plt.ylabel('PDF de ybarra')
plt.plot(y5,pdfy5,color='green')
plt.show()

plt.title('Amostras T n=5')
plt.xlabel('tbarra n=5')
plt.ylabel('Frequência relativa')
plt.hist(tbarra5, weights=np.zeros_like(tbarra5)+1.0/somafat5,bins= 100, rwidth=0.5, color='purple', label='Histograma de tbarra')
plt.show()
plt.title('Amostras T n=5')
plt.xlabel('tbarra n=5')
plt.ylabel('PDF de tbarra')
plt.plot(t5,pdft5,color='purple')
plt.show()


plt.title('Amostras Q n=10')
plt.xlabel('qbarra n=10')
plt.ylabel('Frequência relativa')
plt.hist(qbarra10, weights=np.zeros_like(qbarra10)+1.0/somafaq10,bins= 100, rwidth=0.5, color='red', label='Histograma de qbarra')
plt.show()
plt.title('Amostras Q n=10')
plt.xlabel('qbarra n=10')
plt.ylabel('PDF de qbarra')
plt.plot(q10,pdfq10,color='red')
plt.show()

plt.title('Amostras X n=10')
plt.xlabel('xbarra n=10')
plt.ylabel('Frequência relativa')
plt.hist(xbarra10, weights=np.zeros_like(xbarra10)+1.0/somafax10,bins= 100, rwidth=0.5, color='blue', label='Histograma de xbarra')
plt.show()
plt.title('Amostras X n=10')
plt.xlabel('xbarra n=10')
plt.ylabel('PDF de xbarra')
plt.plot(x10,pdfx10,color='blue')
plt.show()

plt.title('Amostras Y n=10')
plt.xlabel('ybarra n=10')
plt.ylabel('Frequência relativa')
plt.hist(ybarra10, weights=np.zeros_like(ybarra10)+1.0/somafay10,bins= 100, rwidth=0.5, color='green', label='Histograma de ybarra')
plt.show()
plt.title('Amostras Y n=10')
plt.xlabel('ybarra n=10')
plt.ylabel('PDF de ybarra')
plt.plot(y10,pdfy10,color='green')
plt.show()

plt.title('Amostras T n=10')
plt.xlabel('tbarra n=10')
plt.ylabel('Frequência relativa')
plt.hist(tbarra10, weights=np.zeros_like(tbarra10)+1.0/somafat10,bins= 100, rwidth=0.5, color='purple', label='Histograma de tbarra')
plt.show()
plt.title('Amostras T n=10')
plt.xlabel('tbarra n=10')
plt.ylabel('PDF de tbarra')
plt.plot(t10,pdft10,color='purple')
plt.show()


plt.title('Amostras Q n=50')
plt.xlabel('qbarra n=50')
plt.ylabel('Frequência relativa')
plt.hist(qbarra50, weights=np.zeros_like(qbarra50)+1.0/somafaq50,bins= 100, rwidth=0.5, color='red', label='Histograma de qbarra')
plt.show()

plt.title('Amostras Q n=50')
plt.xlabel('qbarra n=50')
plt.ylabel('PDF de qbarra')
plt.plot(q50,pdfq50,color='red')
plt.show()

plt.title('Amostras X n=50')
plt.xlabel('xbarra n=50')
plt.ylabel('Frequência relativa')
plt.hist(xbarra50, weights=np.zeros_like(xbarra50)+1.0/somafax50,bins= 100, rwidth=0.5, color='blue', label='Histograma de xbarra')
plt.show()
plt.title('Amostras X n=50')
plt.xlabel('xbarra n=50')
plt.ylabel('PDF de xbarra')
plt.plot(x50,pdfx50,color='blue')
plt.show()

plt.title('Amostras Y n=50')
plt.xlabel('ybarra n=50')
plt.ylabel('Frequência relativa')
plt.hist(ybarra50, weights=np.zeros_like(ybarra50)+1.0/somafay50,bins= 100, rwidth=0.5, color='green', label='Histograma de ybarra')
plt.show()
plt.title('Amostras Y n=50')
plt.xlabel('ybarra n=50')
plt.ylabel('PDF de ybarra')
plt.plot(y50,pdfy50,color='green')
plt.show()

plt.title('Amostras T n=50')
plt.xlabel('tbarra n=50')
plt.ylabel('Frequência relativa')
plt.hist(tbarra50, weights=np.zeros_like(tbarra50)+1.0/somafat50,bins= 100, rwidth=0.5, color='purple', label='Histograma de tbarra')
plt.show()
plt.title('Amostras T n=50')
plt.xlabel('tbarra n=50')
plt.ylabel('PDF de tbarra')
plt.plot(t50[0:M],pdft50,color='purple')
plt.show()




### pdfs esperadas (variancias) ###

Kq5=min(s2q5)
Kx5=min(s2x5)
Ky5=min(s2y5)
Kt5=min(s2t5)

Mq5=max(s2q5)
Mx5=max(s2x5)+15
My5=max(s2y5)
Mt5=max(s2t5)

Kq10=min(s2q10)
Kx10=min(s2x10)
Ky10=min(s2y10)
Kt10=min(s2t10)

Mq10=max(s2q10)+5
Mx10=max(s2x10)+25
My10=max(s2y10)
Mt10=max(s2t10)+15

Kq50=min(s2q50)+15
Kx50=min(s2x50)+15
Ky50=min(s2y50)
Kt50=min(s2t50)+15

Mq50=max(s2q50)+75
Mx50=max(s2x50)+75
My50=max(s2y50)
Mt50=max(s2t50)+75

q5=np.arange(Kq5,Mq5,(Mq5-Kq5)/M)
q10=np.arange(Kq10,Mq10,(Mq10-Kq10)/M)
q50=np.arange(Kq50,Mq50,(Mq50-Kq50)/M)

x5=np.arange(Kx5,Mx5,(Mx5-Kx5)/M)
x10=np.arange(Kx10,Mx10,(Mx10-Kx10)/M)
x50=np.arange(Kx50,Mx50,(Mx50-Kx50)/M)

y5=np.arange(Ky5,My5,(My5-Ky5)/M)
y10=np.arange(Ky10,My10,(My10-Ky10)/M)
y50=np.arange(Ky50,My50,(My50-Ky50)/M)

t5=np.arange(Kt5,Mt5,(Mt5-Kt5)/M)
t10=np.arange(Kt10,Mt10,(Mt10-Kt10)/M)
t50=np.arange(Kt50,Mt50,(Mt50-Kt50)/M)


gl5=n5-2  #parametros estimados: media e variancia
gl10=n10-2
gl50=n50-2

qqs2q5=np.zeros(M)

qqs2q10=np.zeros(M)

qqs2q50=np.zeros(M)

Gammaq5=0.0

for i in range(M):
    Gammaq5=Gammaq5+(q5[i]**(gl5-1))*np.exp(-q5[i])
    
for i in range(M):
    qqs2q5[i]=((1.0/(2.0**(gl5/2.0)))*Gammaq5)*(q5[i]**((gl5/2.0)-1.0))*np.exp(-q5[i]/2.0)

Gammaq10=0.0

for i in range(M):
    Gammaq10=Gammaq10+(q10[i]**(gl10-1))*np.exp(-q10[i])
    
for i in range(M):
    qqs2q10[i]=((1.0/(2.0**(gl10/2.0)))*Gammaq10)*(q10[i]**((gl10/2.0)-1.0))*np.exp(-q10[i]/2.0)

Gammaq50=0.0

for i in range(M):
    Gammaq50=Gammaq50+(q50[i]**(gl50-1))*np.exp(-q50[i])

for i in range(M):
    qqs2q50[i]=((1.0/(2.0**(gl50/2.0)))*Gammaq50)*(q50[i]**((gl50/2.0)-1.0))*np.exp(-q50[i]/2.0)



qqs2x5=np.zeros(M)

qqs2x10=np.zeros(M)

qqs2x50=np.zeros(M)


Gammax5=0.0

for i in range(M):
    Gammax5=Gammax5+(x5[i]**(gl5-1))*np.exp(-x5[i])

for i in range(M):
    qqs2x5[i]=((1.0/(2.0**(gl5/2.0)))*Gammax5)*(x5[i]**((gl5/2.0)-1.0))*np.exp(-x5[i]/2.0)

Gammax10=0.0

for i in range(M):
    Gammax10=Gammax10+(x10[i]**(gl10-1))*np.exp(-x10[i])
    
for i in range(M):
    qqs2x10[i]=((1.0/(2.0**(gl10/2.0)))*Gammax10)*(x10[i]**((gl10/2.0)-1.0))*np.exp(-x10[i]/2.0)

Gammax50=0.0

for i in range(M):
    Gammax50=Gammax50+(x50[i]**(gl50-1))*np.exp(-x50[i])

for i in range(M):
    qqs2x50[i]=((1.0/(2.0**(gl50/2.0)))*Gammax50)*(x50[i]**((gl50/2.0)-1.0))*np.exp(-x50[i]/2.0)



qqs2y5=np.zeros(M)

qqs2y10=np.zeros(M)

qqs2y50=np.zeros(M)


Gammay5=0.0

for i in range(M):
    Gammay5=Gammay5+(y5[i]**(gl5-1))*np.exp(-y5[i])

for i in range(M):
    qqs2y5[i]=((1.0/(2.0**(gl5/2.0)))*Gammay5)*(y5[i]**((gl5/2.0)-1.0))*np.exp(-y5[i]/2.0)

Gammay10=0.0

for i in range(M):
    Gammay10=Gammay10+(y10[i]**(gl10-1))*np.exp(-y10[i])
    
for i in range(M):
    qqs2y10[i]=((1.0/(2.0**(gl10/2.0)))*Gammay10)*(y10[i]**((gl10/2.0)-1.0))*np.exp(-y10[i]/2.0)

Gammay50=0.0

for i in range(M):
    Gammay50=Gammay50+(y50[i]**(gl50-1))*np.exp(-y50[i])

for i in range(M):
    qqs2y50[i]=((1.0/(2.0**(gl50/2.0)))*Gammay50)*(y50[i]**((gl50/2.0)-1.0))*np.exp(-y50[i]/2.0)



qqs2t5=np.zeros(M)

qqs2t10=np.zeros(M)

qqs2t50=np.zeros(M)


Gammat5=0.0

for i in range(M):
    Gammat5=Gammat5+(t5[i]**(gl5-1))*np.exp(-t5[i])

for i in range(M):
    qqs2t5[i]=((1.0/(2.0**(gl5/2.0)))*Gammat5)*(t5[i]**((gl5/2.0)-1.0))*np.exp(-t5[i]/2.0)

Gammat10=0.0

for i in range(M):
    Gammat10=Gammat10+(t10[i]**(gl10-1))*np.exp(-t10[i])
    
for i in range(M):
    qqs2t10[i]=((1.0/(2.0**(gl10/2.0)))*Gammat10)*(t10[i]**((gl10/2.0)-1.0))*np.exp(-t10[i]/2.0)

Gammat50=0.0

for i in range(M):
    Gammat50=Gammat50+(t50[i]**(gl50-1))*np.exp(-t50[i])

for i in range(M):
    qqs2t50[i]=((1.0/(2.0**(gl50/2.0)))*Gammat50)*(t50[i]**((gl50/2.0)-1.0))*np.exp(-t50[i]/2.0)


### graficos ###

## variancias ##

qqs2q5[:]=qqs2q5[:]/sum(qqs2q5)
plt.title('Amostras Q n=5')
plt.xlabel('s2q n=5')
plt.ylabel('Frequência relativa')
plt.hist(s2q5, weights=np.zeros_like(s2q5)+1.0/somafaqs25,bins= 100, rwidth=0.5, color='red', label='Histograma de s2q')
plt.show()
plt.title('Amostras Q n=5')
plt.xlabel('s2q n=5')
plt.ylabel('PDF s2q')
plt.plot(q5,qqs2q5,color='red')
plt.show()

qqs2x5[:]=qqs2x5[:]/sum(qqs2x5)
plt.title('Amostras X n=5')
plt.xlabel('s2x n=5')
plt.ylabel('Frequência relativa')
plt.hist(s2x5, weights=np.zeros_like(s2x5)+1.0/somafaxs25,bins= 100, rwidth=0.5, color='blue', label='Histograma de s2x')
plt.show()
plt.title('Amostras X n=5')
plt.xlabel('s2x n=5')
plt.ylabel('PDF s2x')
plt.plot(x5,qqs2x5,color='blue')
plt.show()

qqs2y5[:]=qqs2y5[:]/sum(qqs2y5)
plt.title('Amostras Y n=5')
plt.xlabel('s2y n=5')
plt.ylabel('Frequência relativa')
plt.hist(s2y5, weights=np.zeros_like(s2y5)+1.0/somafays25,bins= 100, rwidth=0.5, color='green', label='Histograma de s2y')
plt.show()
plt.title('Amostras Y n=5')
plt.xlabel('s2y n=5')
plt.ylabel('PDF s2y')
plt.plot(y5,qqs2y5,color='green')
plt.show()

qqs2t5[:]=qqs2t5[:]/sum(qqs2t5)
plt.title('Amostras T n=5')
plt.xlabel('s2t n=5')
plt.ylabel('Frequência relativa')
plt.hist(s2t5, weights=np.zeros_like(s2t5)+1.0/somafats25,bins= 100, rwidth=0.5, color='purple', label='Histograma de s2t')
plt.show()
plt.title('Amostras T n=5')
plt.xlabel('s2t n=5')
plt.ylabel('PDF s2t')
plt.plot(t5,qqs2t5,color='purple')
plt.show()

qqs2q10[:]=qqs2q10[:]/sum(qqs2q10)
plt.title('Amostras Q n=10')
plt.xlabel('s2q n=10')
plt.ylabel('Frequência relativa')
plt.hist(s2q10, weights=np.zeros_like(s2q10)+1.0/somafaqs210,bins= 100, rwidth=0.5, color='red', label='Histograma de s2q')
plt.show()
plt.title('Amostras Q n=10')
plt.xlabel('s2q n=10')
plt.ylabel('PDF s2q')
plt.plot(q10,qqs2q10,color='red')
plt.show()

qqs2x10[:]=qqs2x10[:]/sum(qqs2x10)
plt.title('Amostras X n=10')
plt.xlabel('s2x n=10')
plt.ylabel('Frequência relativa')
plt.hist(s2x10, weights=np.zeros_like(s2x10)+1.0/somafaxs210,bins= 100, rwidth=0.5, color='blue', label='Histograma de s2x')
plt.show()
plt.title('Amostras X n=10')
plt.xlabel('s2x n=10')
plt.ylabel('PDF s2x')
plt.plot(x10,qqs2x10,color='blue')
plt.show()

qqs2y10[:]=qqs2y10[:]/sum(qqs2y10)
plt.title('Amostras Y n=10')
plt.xlabel('s2y n=10')
plt.ylabel('Frequência relativa')
plt.hist(s2y10, weights=np.zeros_like(s2y10)+1.0/somafays210,bins= 100, rwidth=0.5, color='green', label='Histograma de s2y')
plt.show()
plt.title('Amostras Y n=10')
plt.xlabel('s2y n=10')
plt.ylabel('PDF s2y')
plt.plot(y10,qqs2y10,color='green')
plt.show()

qqs2t10[:]=qqs2t10[:]/sum(qqs2t10)
plt.title('Amostras T n=10')
plt.xlabel('s2t n=10')
plt.ylabel('Frequência relativa')
plt.hist(s2t10, weights=np.zeros_like(s2t10)+1.0/somafats210,bins= 100, rwidth=0.5, color='purple', label='Histograma de s2t')
plt.show()
plt.title('Amostras T n=10')
plt.xlabel('s2t n=10')
plt.ylabel('PDF s2t')
plt.plot(t10,qqs2t10,color='purple')
plt.show()

qqs2q50[:]=qqs2q50[:]/sum(qqs2q50)
plt.title('Amostras Q n=50')
plt.xlabel('s2q n=50')
plt.ylabel('Frequência relativa')
plt.hist(s2q50, weights=np.zeros_like(s2q50)+1.0/somafaqs250,bins= 100, rwidth=0.5, color='red', label='Histograma de s2q')
plt.show()
plt.title('Amostras Q n=50')
plt.xlabel('s2q n=50')
plt.ylabel('PDF s2q')
plt.plot(q50,qqs2q50,color='red')
plt.show()

qqs2x50[:]=qqs2x50[:]/sum(qqs2x50)
plt.title('Amostras X n=50')
plt.xlabel('s2x n=50')
plt.ylabel('Frequência relativa')
plt.hist(s2x50, weights=np.zeros_like(s2x50)+1.0/somafaxs250,bins= 100, rwidth=0.5, color='blue', label='Histograma de s2x')
plt.show()
plt.title('Amostras X n=50')
plt.xlabel('s2x n=50')
plt.ylabel('PDF s2x')
plt.plot(x50,qqs2x50,color='blue')
plt.show()

qqs2y50[:]=qqs2y50[:]/sum(qqs2y50)
plt.title('Amostras Y n=50')
plt.xlabel('s2y n=50')
plt.ylabel('Frequência relativa')
plt.hist(s2y50, weights=np.zeros_like(s2y50)+1.0/somafays250,bins= 100, rwidth=0.5, color='green', label='Histograma de s2y')
plt.show()
plt.title('Amostras Y n=50')
plt.xlabel('s2y n=50')
plt.ylabel('PDF s2y')
plt.plot(y50,qqs2y50,color='green')
plt.show()

qqs2t50[:]=qqs2t50[:]/sum(qqs2t50)
plt.title('Amostras T n=50')
plt.xlabel('s2t n=50')
plt.ylabel('Frequência relativa')
plt.hist(s2t50, weights=np.zeros_like(s2t50)+1.0/somafats250,bins= 100, rwidth=0.5, color='purple', label='Histograma de s2t')
plt.show()
plt.title('Amostras T n=50')
plt.xlabel('s2t n=50')
plt.ylabel('PDF s2t')
plt.plot(t50,qqs2t50,color='purple')
plt.show()
