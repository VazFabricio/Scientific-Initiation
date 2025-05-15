import scipy.io
import pandas as pd
import numpy as np
from scipy import sparse

# Parametros ajustaveis manualmente
nfp=5;             #Funcoes de pertinencia
alfa=0.01;         #Alfa - parametro de atualizacao
nepoca=5;        #Numero de epocas de treinamento dos pesos para cada geração da PG

#Parametros Programacao Genetica
tamPop=50;
numGeracoes=100;
taxaCruza=0.7;
taxaMuta=0.08;
nfpMax=7; # maximo de funções de pertinência geradas em cada indivíduo

_xt_temp = np.loadtxt('d:\\ScientificIni\\Scientific-Initiation\\xt.csv', delimiter=',', skiprows=1)
xt = _xt_temp[:, 1:] 

_yt_temp = np.loadtxt('d:\\ScientificIni\\Scientific-Initiation\\yt.csv', delimiter=',', skiprows=1)
yt = _yt_temp[:, 1:]

npt, nin = (xt).shape

nptTr = round(npt*0.6);
nptVal = round(npt*0.2);

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
for i in range(0,nfp):
    for j in range(0,nin):
        p[j, i] = np.random.rand()

    q[i] = np.random.rand()

