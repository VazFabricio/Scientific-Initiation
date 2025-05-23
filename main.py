import scipy.io
import pandas as pd
import numpy as np
import saida
import skfuzzy as fuzz
from scipy import sparse
import time
import matplotlib.pyplot as plt

start_time = time.time() 

# Parametros ajustaveis manualmente
nfp=5             #Funcoes de pertinencia
alfa=0.01         #Alfa - parametro de atualizacao
nepoca=5       #Numero de epocas de treinamento dos pesos para cada geração da PG

#Parametros Programacao Genetica
tamPop=2
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

npt=nptTr

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

#Funcao de pertinencia gaussiana
xval = np.linspace(xmin[0], xmax[0], npt)
print(xval)

xval[0] = xval[0]/10
xval[1] = xval[1]/10

# plt.figure(2)
# for j in range(nfp):
#     for i in range(nin):
#         w = fuzz.gaussmf(xval, c[i, j], s[i, j])
#         ax = plt.subplot(2, nin, i + 1)
#         ax.plot(xval, w, 'k')
#         ax.grid(True)
#         ax.set_title('Membership Functions - Initial')
#         ax.set_xlabel(f'$X{{{i + 1}}}$')  # LaTeX-style subscript
#         ax.set_ylabel('Membership')
#         ax.set_xlim([xmin[i], xmax[i]])

# plt.tight_layout()
# plt.show()

ytst = saida.saida(xt,c,s,p,q,nfp)
yvst = saida.saida(xv,c,s,p,q,nfp)

erro = []
yst = ytst

for i in range(numGeracoes):
    print('Geração ', i)

    erro[i] = (0.5 * np.sum((yst - ydt)**2)) / npt
    dyjdqj = 1

    for j in range(nepoca):
        for k in range(npt):
            ys, w, y, b = saida(xt[k,:], c, s, p, q, nfp)
            dedys = (ys - ydt[k])
            for l in range(nfp):
                dysdyj = w[l] /b
                dysdwj = (y[l] - ys) /b
                for m in range(nin):
                    dyjdpj = xt[k, m]

                    p[m, j] -= ((alfa/10.0)*dedys*dysdyj*dyjdpj)

                q[j] -= ((alfa/10.0)*dedys*dysdyj*dyjdqj)

    # Finalização das epocas - Linha: 199
    pop

end_time = time.time() 
execution_time = end_time - start_time 
print(f"The script took {execution_time:.4f} seconds to execute.")