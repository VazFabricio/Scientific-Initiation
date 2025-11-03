import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import saida
import gerarnovapop

start_time = time.time()

# -------------------------
# Parâmetros ajustáveis
# -------------------------
nfp = 5             # Funções de pertinência por variável
nepoca = 5          # (não usado quando não há gradiente)
# Parâmetros Programação Genética
tamPop = 50
numGeracoes = 25
taxaCruza = 0.7
taxaMuta = 0.08
nfpMax = 7

# -------------------------
# Carregar dados
# -------------------------
_xt_temp = np.loadtxt('xt.csv', delimiter=',', skiprows=1)
xt_all = _xt_temp[:, 1:]
_yt_temp = np.loadtxt('yt.csv', delimiter=',', skiprows=1)
yt_all = _yt_temp[:, 1:]

# garantir 1D para yt
yt_all = yt_all.ravel()

# -------------------------
# Partição de treino / validação
# -------------------------
npt_total, nin = xt_all.shape
nptTr = int(round(npt_total * 0.6))

x_train = xt_all[:nptTr, :]
x_val = xt_all[nptTr:, :]

y_train = yt_all[:nptTr]
y_val = yt_all[nptTr:]

npt = x_train.shape[0]

xmin = x_train.min(axis=0)
xmax = x_train.max(axis=0)
delta = (xmax - xmin) / (nfp - 1)

aux = sparse.csr_matrix(x_train)

# -------------------------
# População inicial
# cada indivíduo tem cs(centros), ss(sigmas), p, q, nfps
# -------------------------
rng = np.random.default_rng()
pop = []

for z in range(tamPop):
    nfpSort = nfp 
    cs = np.empty((nin, nfpSort), dtype=float)
    ss = np.empty((nin, nfpSort), dtype=float)

    # p e q individuais (cada indivíduo tem seus próprios p e q)
    p_ind = rng.random((nin, nfpSort))
    q_ind = rng.random(nfpSort)

    for j in range(nfpSort):
        for i in range(nin):
            cs[i, j] = xmin[i] + rng.random() * (xmax[i] - xmin[i])
            ss[i, j] = rng.random() * (xmax[i] - xmin[i])

    indiv = {
        'nfps': nfpSort,
        'cs': cs,
        'ss': ss,
        'p': p_ind,
        'q': q_ind
    }

    # chamada a saida usando os parâmetros do indivíduo
    out = saida.saida(x_train, cs, ss, p_ind, q_ind, nfpSort)
    y_pred = np.asarray(out[0] if isinstance(out, (list, tuple)) else out).ravel()
    indiv['saida'] = y_pred
    indiv['fitness'] = (0.5 * np.sum((y_pred - y_train) ** 2)) / npt

    pop.append(indiv)

# melhor indivíduo inicial
melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

# parâmetros do melhor
c = np.asarray(pop[melhorindv]['cs'], dtype=float)
s = np.asarray(pop[melhorindv]['ss'], dtype=float)
nfp = pop[melhorindv]['nfps']

# xval para plot das MF (apenas 1D para visualização)
xval = np.linspace(xmin[0], xmax[0], npt)
if xval.size > 1:
    xval[0] = xval[0] / 10.0
    xval[1] = xval[1] / 10.0

# Saídas iniciais
ytst_out = saida.saida(x_train, c, s, pop[melhorindv]['p'], pop[melhorindv]['q'], nfp)
ytst = np.asarray(ytst_out[0] if isinstance(ytst_out, (list, tuple)) else ytst_out).ravel()
yvst_out = saida.saida(x_val, c, s, pop[melhorindv]['p'], pop[melhorindv]['q'], nfp)
yvst = np.asarray(yvst_out[0] if isinstance(yvst_out, (list, tuple)) else yvst_out).ravel()

# -------------------------
# Loop da Programação Genética (sem gradiente)
# -------------------------
erro = []
yst = ytst.copy()

for gen in range(numGeracoes):
    print(f'Geração {gen+1}')
    # erro calculado com a saída atual do melhor conjunto de parâmetros
    erro.append((0.5 * np.sum((yst - y_train) ** 2)) / npt)

    # gera nova população
    pop = gerarnovapop.gerarnovapop(pop, melhorindv, tamPop, taxaCruza, taxaMuta, xmax, xmin)

    # avaliar cada indivíduo (usar p e q do indivíduo)
    for z in range(tamPop):
        indiv = pop[z]
        out = saida.saida(x_train, indiv['cs'], indiv['ss'], indiv['p'], indiv['q'], indiv['nfps'])
        y_pred = np.asarray(out[0] if isinstance(out, (list, tuple)) else out).ravel()
        indiv['saida'] = y_pred
        indiv['fitness'] = (0.5 * np.sum((y_pred - y_train) ** 2)) / npt

    # atualizar melhor indivíduo
    melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

    # atualizar parâmetros com o melhor
    c = np.asarray(pop[melhorindv]['cs'], dtype=float)
    s = np.asarray(pop[melhorindv]['ss'], dtype=float)
    nfp = pop[melhorindv]['nfps']

    # recomputar yst para o próximo cálculo de erro
    out_best = saida.saida(x_train, c, s, pop[melhorindv]['p'], pop[melhorindv]['q'], nfp)
    yst = np.asarray(out_best[0] if isinstance(out_best, (list, tuple)) else out_best).ravel()

# erro final
erro.append((0.5 * np.sum((yst - y_train) ** 2)) / npt)

# Resultado final - treino e validação
out_final_train = saida.saida(x_train, c, s, pop[melhorindv]['p'], pop[melhorindv]['q'], nfp)
yst = np.asarray(out_final_train[0] if isinstance(out_final_train, (list, tuple)) else out_final_train).ravel()
out_final_val = saida.saida(x_val, c, s, pop[melhorindv]['p'], pop[melhorindv]['q'], nfp)
ysv = np.asarray(out_final_val[0] if isinstance(out_final_val, (list, tuple)) else out_final_val).ravel()

end_time = time.time()
print(f"The script took {end_time - start_time:.3f} seconds to execute.")

# -------------------------
# PLOTS FINAIS (MFs, Erros, Saídas)
# -------------------------
import skfuzzy as fuzz

# -------- 1. Membership Functions (Inicial e Final) --------
fig_mf, axes = plt.subplots(2, nin, figsize=(4 * nin, 6))

# MF Iniciais
for j in range(nfp):
    for i in range(nin):
        w_init = fuzz.gaussmf(xval, c[i, j], s[i, j])
        ax = axes[0, i] if nin > 1 else axes
        ax.plot(xval, w_init, color='gray', linestyle='--', linewidth=1.0)
        if j == 0:
            ax.set_title(f'Membership Functions - Initial (X{i+1})')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)

# MF Finais
for j in range(nfp):
    for i in range(nin):
        w_final = fuzz.gaussmf(xval, c[i, j], s[i, j])
        ax = axes[1, i] if nin > 1 else axes
        ax.plot(xval, w_final, color='black', linewidth=1.2)
        if j == 0:
            ax.set_title(f'Membership Functions - Final (X{i+1})')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)

fig_mf.tight_layout()

# -------- 2. Training Error --------
fig_err = plt.figure()
plt.plot(erro, 'r', linewidth=1.5)
plt.xlabel('Geração')
plt.ylabel('EQM')
plt.title('Erro Quadrático Médio')
plt.grid(True)

# -------- 3. Training and Validation Outputs --------
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].plot(y_train, 'r', label='Saída Desejada')
axs[0, 0].plot(ytst, 'k', label='Saída Inicial')
axs[0, 0].set_title('Treinamento - Saída Desejada x Saída Inicial')
axs[0, 0].set_xlabel('Pontos')
axs[0, 0].set_ylabel('Y')
axs[0, 0].legend()

axs[0, 1].plot(yst, 'g')
axs[0, 1].set_title('Treinamento - Saída Final')
axs[0, 1].set_xlabel('Pontos')
axs[0, 1].set_ylabel('Y')

axs[0, 2].plot(y_train, 'r', label='Saída Desejada')
axs[0, 2].plot(yst, 'g', label='Saída Final')
axs[0, 2].set_title('Treinamento - Saída Desejada x Saída Final')
axs[0, 2].set_xlabel('Pontos')
axs[0, 2].set_ylabel('Y')
axs[0, 2].legend()

axs[1, 0].plot(y_val, 'r', label='Saída Desejada')
axs[1, 0].plot(yvst, 'k', label='Saída Inicial')
axs[1, 0].set_title('Validação - Saída Desejada x Saída Inicial')
axs[1, 0].set_xlabel('Pontos')
axs[1, 0].set_ylabel('Y')
axs[1, 0].legend()

axs[1, 1].plot(ysv, 'g')
axs[1, 1].set_title('Validação - Saída Final')
axs[1, 1].set_xlabel('Pontos')
axs[1, 1].set_ylabel('Y')

axs[1, 2].plot(y_val, 'r', label='Saída Desejada')
axs[1, 2].plot(ysv, 'g', label='Saída Final')
axs[1, 2].set_title('Validação - Saída Desejada x Saída Final')
axs[1, 2].set_xlabel('Pontos')
axs[1, 2].set_ylabel('Y')
axs[1, 2].legend()

fig3.tight_layout()

# -------- 4. Final Training Output --------
fig4 = plt.figure()
plt.plot(y_train, 'r', label='Saída Desejada')
plt.plot(yst, 'b', label='Saída GP-NFN-I')
plt.xlabel('Pontos')
plt.ylabel('Y')
plt.title('Treinamento - Saída Desejada x Saída Final')
plt.legend()
plt.grid(True)

# -------- 5. Final Validation Output --------
fig5 = plt.figure()
plt.plot(y_val, 'r', label='Desired Output')
plt.plot(ysv, 'g', label='Estimated Output')
plt.xlabel('Amostras')
plt.ylabel('Y')
plt.title('Validation - Desired Output x Estimated Output')
plt.legend()
plt.grid(True)

plt.show()