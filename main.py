import scipy.io
import pandas as pd
import numpy as np
import saida
import skfuzzy as fuzz
from scipy import sparse
import time
start_time = time.time() 

# Parametros ajustaveis manualmente
nfp=5             #Funcoes de pertinencia
alfa=0.01         #Alfa - parametro de atualizacao
nepoca=5       #Numero de epocas de treinamento dos pesos para cada geração da PG

#Parametros Programacao Genetica
tamPop=50
numGeracoes=100
taxaCruza=0.7
taxaMuta=0.08
nfpMax=7; # maximo de funções de pertinência geradas em cada indivíduo

_xt_temp = np.loadtxt('xt.csv', delimiter=',', skiprows=1)
xt = _xt_temp[:, 1:] 
_yt_temp = np.loadtxt('yt.csv', delimiter=',', skiprows=1)
yt = _yt_temp[:, 1:]

npt, nin = (xt).shape

nptTr = round(npt*0.6)
nptVal = round(npt*0.2)

xv = xt[nptTr-1:]
xt = xt[:nptTr]

ydv=yt[nptTr-1:]
ydt=yt[:nptTr]

xmin=xt.min(axis=0)
xmax=xt.max(axis=0)

delta=(xmax-xmin) / (nfp-1)

aux = sparse.csr_matrix(xt)

#Calculo dos parametros
p = np.empty((nin, nfp))
q = np.empty(nfp)
for i in range(nfp):
    for j in range(nin):
        p[j, i] = np.random.rand()

    q[i] = np.random.rand()
    
# #Gerar população inicial
pop = []

for z in range(tamPop):
    nfpSort = nfp
    cs = np.empty((nin, nfpSort))
    ss = np.empty((nin, nfpSort))
    for i in range(nfpSort):
        for j in range(nin):
            cs[j, i] = xmin[j] + np.random.rand()*(xmax[j]-xmin[j])
            ss[j, i] = np.random.rand()*(xmax[j]-xmin[j])

    individuo = {
        'nfps': nfpSort,
        'cs': cs,
        'ss': ss
    }
    individuo['saida'] = saida.saida(xt, individuo['cs'], individuo['ss'], p, q, individuo['nfps'])
    individuo['fitness'] = (0.5 * np.sum((individuo['saida'] - ydt)**2)) / npt
    pop.append(individuo)

    if z == 0:
        melhorindv = z
    else:
        if individuo['fitness'] < pop[melhorindv]['fitness']:
            melhorindv = z
    
c = pop[melhorindv]['cs']
s = pop[melhorindv]['ss']
nfp = pop[melhorindv]['nfps']
auxsize = tamPop
novapop = pop






end_time = time.time() 
execution_time = end_time - start_time 
print(f"The script took {execution_time:.4f} seconds to execute.")