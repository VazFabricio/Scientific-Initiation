import scipy.io
import pandas as pd
import numpy as np
import saida
import gerarnovapop
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
tamPop=50
numGeracoes=25
taxaCruza=0.7
taxaMuta=0.08
nfpMax=7; 

_xt_temp = np.loadtxt('xt.csv', delimiter=',', skiprows=1)
xt = _xt_temp[:, 1:] 
_yt_temp = np.loadtxt('yt.csv', delimiter=',', skiprows=1)
yt = _yt_temp[:, 1:]

# temp = np.loadtxt('AirfoilSelfNoise.csv', delimiter=',', skiprows=1)
# xt = temp[:, :-1] 
# yt = temp[:, -1]  

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

p = np.empty((nin, nfp))
q = np.empty(nfp)
for i in range(nfp):
    for j in range(nin):
        p[j, i] = np.random.rand()

    q[i] = np.random.rand()

pop = []

for z in range(tamPop):
    nfpSort = nfp
    cs = np.empty((nin, nfpSort))
    ss = np.empty((nin, nfpSort))
    
    #Criar p e q individual: 
    p_ind = np.random.rand(nin, nfpSort)
    q_ind = np.random.rand(nfpSort)
    
    for i in range(nfpSort):
        for j in range(nin):
            cs[j, i] = xmin[j] + np.random.rand()*(xmax[j]-xmin[j])
            ss[j, i] = np.random.rand()*(xmax[j]-xmin[j])

    individuo = {
        'nfps': nfpSort,
        'cs': cs,
        'ss': ss,
        'p': p_ind,
        'q': q_ind
    }
    
    individuo['saida'], _, _, _ = saida.saida(xt, individuo['cs'], individuo['ss'], individuo['p'], individuo['q'], individuo['nfps'])
    individuo['fitness'] = (0.5 * np.sum((individuo['saida'] - ydt)**2)) / npt
    pop.append(individuo)

    if z == 0:
        melhorindv = z
    else:
        if individuo['fitness'] < pop[melhorindv]['fitness']:
            melhorindv = z
    
c = np.array(pop[melhorindv]['cs'], dtype=float)
s = np.array(pop[melhorindv]['ss'], dtype=float)
s = pop[melhorindv]['ss']
nfp = pop[melhorindv]['nfps']
auxsize = tamPop
novapop = pop

#Funcao de pertinencia gaussiana
xval = np.linspace(xmin[0], xmax[0], npt)

xval[0] = xval[0]/10
xval[1] = xval[1]/10

# fig1, axes = plt.subplots(2, nin, figsize=(4 * nin, 6))

# Membership Functions - Initial (top row)
# for j in range(nfp):
#     for i in range(nin):
#         w_init = fuzz.gaussmf(xval, c[i, j], s[i, j])
#         ax = axes[0, i] if nin > 1 else axes
#         ax.plot(xval, w_init, 'gray', linestyle='--', label='Initial' if j == 0 else "")
#         ax.set_title(f'Membership Functions - Initial (X{i+1})')
#         ax.set_xlabel(f'X_{i+1}')
#         ax.set_ylabel('Membership')
#         ax.set_xlim(xmin[i], xmax[i])
#         ax.grid(True)


ytst, _, _, _ = saida.saida(xt,c,s,p,q,nfp)
yvst, _, _, _ = saida.saida(xv,c,s,p,q,nfp)

erro = []
yst = ytst

for i in range(numGeracoes):
    print('Geração ', i + 1)

    erro.append((0.5 * np.sum((yst - ydt)**2)) / npt)
    

    pop = gerarnovapop.gerarnovapop(pop, melhorindv, tamPop, taxaCruza, taxaMuta, xmax, xmin)

    for z in range(tamPop):

        pop[z]['saida'] = saida.saida(xt, pop[z]['cs'], pop[z]['ss'], pop[z]['p'], pop[z]['q'], pop[z]['nfps'])[0]
        pop[z]['fitness'] = (0.5 * np.sum((pop[z]['saida'] - ydt) ** 2)) / npt
        
        if z == 0:
            melhorindv = z
        else:
            if pop[z]['fitness'] < pop[melhorindv]['fitness']:
                melhorindv = z

    c = pop[melhorindv]['cs']
    s = pop[melhorindv]['ss']
    nfp = pop[melhorindv]['nfps']

    p = pop[melhorindv]['p']
    q = pop[melhorindv]['q']
    
    yst = saida.saida(xt, c, s, p, q, nfp)[0]

erro.append((0.5 * np.sum((yst - ydt) ** 2)) / npt)
yst = saida.saida(xt, c, s, p, q, nfp)[0] 
ysv = saida.saida(xv, c, s, p, q, nfp)[0] 

end_time = time.time() 
execution_time = end_time - start_time 
print(f"The script took {execution_time:.4f} seconds to execute.")

c = np.asarray(pop[melhorindv]['cs'])
s = np.asarray(pop[melhorindv]['ss'])

# fig1, axes = plt.subplots(2, nin, figsize=(4 * nin, 6))
# for j in range(nfp):
#     for i in range(nin):
#         w = fuzz.gaussmf(xval, c[i, j], s[i, j])
#         ax = axes[1, i] if nin > 1 else axes
#         ax.plot(xval, w, 'k')
#         ax.set_title('Membership Functions - Final')
#         ax.set_xlabel(f'X_{i+1}')
#         ax.set_ylabel('Membership')
#         ax.set_xlim(xmin[i], xmax[i])
#         ax.grid(True)

# fig1.tight_layout()

# ---------- 2. Training Error (MSE) ----------
fig2 = plt.figure()
plt.plot(erro, 'r', linewidth=1.5)
plt.xlabel('Geração')
plt.ylabel('EQM')
plt.title('Erro Quadrático Médio')
plt.grid(True)

# # ---------- 3. Training and Validation Outputs ----------
# fig3, axs = plt.subplots(2, 3, figsize=(15, 8))

# axs[0, 0].plot(ydt, 'r', label='Saída Desejada')
# axs[0, 0].plot(ytst, 'k', label='Saída Inicial')
# axs[0, 0].set_title('Treinamento - Saída Desejada x Saída Inicial')
# axs[0, 0].set_xlabel('Pontos')
# axs[0, 0].set_ylabel('X')
# axs[0, 0].legend()

# axs[0, 1].plot(yst, 'g')
# axs[0, 1].set_title('Treinamento - Saída Final')
# axs[0, 1].set_xlabel('Pontos')
# axs[0, 1].set_ylabel('X')

# axs[0, 2].plot(ydt, 'r', label='Saída Desejada')
# axs[0, 2].plot(yst, 'g', label='Saída Final')
# axs[0, 2].set_title('Treinamento - Saída Desejada x Saída Final')
# axs[0, 2].set_xlabel('Pontos')
# axs[0, 2].set_ylabel('Y')
# axs[0, 2].legend()

# axs[1, 0].plot(ydv, 'r', label='Saída Desejada')
# axs[1, 0].plot(yvst, 'k', label='Saída Inicial')
# axs[1, 0].set_title('Validação - Saída Desejada x Saída Inicial')
# axs[1, 0].set_xlabel('Pontos')
# axs[1, 0].set_ylabel('X')
# axs[1, 0].legend()

# axs[1, 1].plot(ysv, 'g')
# axs[1, 1].set_title('Validação - Saída Final')
# axs[1, 1].set_xlabel('Pontos')
# axs[1, 1].set_ylabel('X')

# axs[1, 2].plot(ydv, 'r', label='Saída Desejada')
# axs[1, 2].plot(ysv, 'g', label='Saída Final')
# axs[1, 2].set_title('Validação - Saída Desejada x Saída Final')
# axs[1, 2].set_xlabel('Pontos')
# axs[1, 2].set_ylabel('Y')
# axs[1, 2].legend()

# fig3.tight_layout()

# # ---------- 4. Final Training Output ----------
# fig4 = plt.figure()
# plt.plot(ydt, 'r', label='Saída Desejada')
# plt.plot(yst, 'b', label='Saída GP-NFN-I')
# plt.xlabel('Pontos')
# plt.ylabel('Y')
# plt.title('Treinamento - Saída Desejada x Saída Final')
# plt.legend()

# ---------- 5. Final Validation Output ----------
fig5 = plt.figure()
plt.plot(ydv, 'r', label='Desired Output')
plt.plot(ysv, 'g', label='Estimated Output')
plt.xlabel('Samples')
plt.ylabel('Y')
plt.title('Validation - Desired Output x Estimated Output')
plt.legend()

# ---------- Show All ----------
plt.show()