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
nfp = 5             # Funções de pertinência
alfa = 0.01         # taxa de aprendizado (alfa)
nepoca = 5          # número de épocas de atualização dos pesos por geração

# Parâmetros Programação Genética
tamPop = 5
numGeracoes = 5
taxaCruza = 0.7
taxaMuta = 0.08
nfpMax = 7          # máximo de FPs geradas por indivíduo

# -------------------------
# Carregar dados
# -------------------------
_xt_temp = np.loadtxt('xt.csv', delimiter=',', skiprows=1)
xt = _xt_temp[:, 1:]
_yt_temp = np.loadtxt('yt.csv', delimiter=',', skiprows=1)
yt = _yt_temp[:, 1:]

# garantir shapes 1D para yt
yt = yt.ravel()

# -------------------------
# Partição de treino/validação
# -------------------------
npt_total, nin = xt.shape
nptTr = int(round(npt_total * 0.6))

x_train = xt[:nptTr, :]
x_val = xt[nptTr:npt_total, :]

y_train = yt[:nptTr]
y_val = yt[nptTr:npt_total]

npt = nptTr 

xmin = x_train.min(axis=0)
xmax = x_train.max(axis=0)
delta = (xmax - xmin) / (nfp - 1)

aux = sparse.csr_matrix(x_train)

# -------------------------
# Inicialização de parâmetros p e q
# p: (nin, nfp), q: (nfp,)
# -------------------------
rng = np.random.default_rng()
p = rng.random((nin, nfp))
q = rng.random(nfp)

# -------------------------
# Função gaussiana
# -------------------------
def gaussmf(x, mean, sigma):
    # evitar divisão por zero
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))

# -------------------------
# Gera população inicial
# cada indivíduo: dict com 'nfps', 'cs' (centros), 'ss' (sigmas), 'saida', 'fitness'
# -------------------------
pop = []
for z in range(tamPop):
    nfpSort = nfp
    cs = np.empty((nin, nfpSort))
    ss = np.empty((nin, nfpSort))
    for j in range(nfpSort):
        for i in range(nin):
            cs[i, j] = xmin[i] + rng.random() * (xmax[i] - xmin[i])
            ss[i, j] = rng.random() * (xmax[i] - xmin[i])  # sigma relativo ao range

    indiv = {'nfps': nfpSort, 'cs': cs, 'ss': ss}
    saida_full = saida.saida(x_train, cs, ss, p, q, nfpSort)
    # pegar primeiro elemento (vetor de saídas)
    if isinstance(saida_full, tuple) or isinstance(saida_full, list):
        y_pred = np.asarray(saida_full[0]).ravel()
    else:
        y_pred = np.asarray(saida_full).ravel()
    indiv['saida'] = y_pred
    indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - y_train) ** 2)) / npt
    pop.append(indiv)

# identifica melhor indivíduo
melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

# inicializar variáveis com melhor indivíduo
c = pop[melhorindv]['cs'].copy()
s = pop[melhorindv]['ss'].copy()
nfp = pop[melhorindv]['nfps']
novapop = pop.copy()

# -------------------------
# Preparar xval para plot das MF
# -------------------------
xval = np.linspace(xmin[0], xmax[0], npt)

# plot - MF iniciais
fig_mf_init, axes = plt.subplots(2, max(1, nin), figsize=(4 * max(1, nin), 6))
for j in range(nfp):
    for i in range(nin):
        # mean = c[i,j], sigma = s[i,j]
        w = gaussmf(xval, c[i, j], s[i, j])
        ax = axes[0, i] if nin > 1 else axes
        ax.plot(xval, w, color='black', linewidth=0.8)
        ax.set_title('Membership Functions - Initial' if j == 0 else '')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)

# -------------------------
# Saída inicial sem treinamento
# -------------------------
ytst_full = saida.saida(x_train, c, s, p, q, nfp)
ytst = np.asarray(ytst_full[0] if isinstance(ytst_full, (list,tuple)) else ytst_full).ravel()
yvst_full = saida.saida(x_val, c, s, p, q, nfp)
yvst = np.asarray(yvst_full[0] if isinstance(yvst_full, (list,tuple)) else yvst_full).ravel()

# -------------------------
# Treinamento (laço de gerações + atualizações de p,q por época)
# -------------------------
erro = []
yst = ytst.copy()

for gen in range(numGeracoes):
    print(f'Geração {gen+1}')
    erro.append((0.5 * np.sum((yst - y_train) ** 2)) / npt)
    dyjdqj = 1.0

    # treinamento dos pesos p e q (gradient-like)
    for _ in range(nepoca):
        for k in range(npt):
            sample = x_train[k, :]
            # chamar saida para 1 amostra
            ys_full = saida.saida(sample, c, s, p, q, nfp)
            if isinstance(ys_full, (list, tuple)):
                ys = float(np.asarray(ys_full[0]).ravel())
                w = np.asarray(ys_full[1]).ravel()
                y_vec = np.asarray(ys_full[2])
                b = float(ys_full[3])
            else:
                ys = float(np.asarray(ys_full).ravel())
                raise RuntimeError("saida.saida não retornou (ys, w, y, b). Ajuste a função 'saida'.")

            dedys = ys - float(y_train[k])  # erro escalar

            for j in range(nfp):
                dysdyj = w[j] / b
                # dysdwj = (y_vec[j] - ys) / b   # não usado diretamente aqui
                for i in range(nin):
                    dyjdpj = sample[i]
                    p[i, j] = p[i, j] - ((alfa / 10.0) * dedys * dysdyj * dyjdpj)
                q[j] = q[j] - ((alfa / 10.0) * dedys * dysdyj * dyjdqj)

    pop = gerarnovapop.gerarnovapop(pop, melhorindv, tamPop, taxaCruza, taxaMuta, xmax, xmin)

    # re-avaliar fitness
    for z in range(tamPop):
        indiv = pop[z]
        saida_full = saida.saida(x_train, indiv['cs'], indiv['ss'], p, q, indiv['nfps'])
        y_pred = np.asarray(saida_full[0] if isinstance(saida_full, (list,tuple)) else saida_full).ravel()
        indiv['saida'] = y_pred
        indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - y_train) ** 2)) / npt

    melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

    # atualizar melhores parâmetros
    c = pop[melhorindv]['cs'].copy()
    s = pop[melhorindv]['ss'].copy()
    nfp = pop[melhorindv]['nfps']
    # recomputar yst
    yst_full = saida.saida(x_train, c, s, p, q, nfp)
    yst = np.asarray(yst_full[0] if isinstance(yst_full, (list,tuple)) else yst_full).ravel()

# erro final
erro.append((0.5 * np.sum((yst - y_train) ** 2)) / npt)

# Resultado final - teste / validação
yst_full = saida.saida(x_train, c, s, p, q, nfp)
yst = np.asarray(yst_full[0] if isinstance(yst_full, (list,tuple)) else yst_full).ravel()
ysv_full = saida.saida(x_val, c, s, p, q, nfp)
ysv = np.asarray(ysv_full[0] if isinstance(ysv_full, (list,tuple)) else ysv_full).ravel()

end_time = time.time()
print(f"The script took {end_time - start_time:.3f} seconds to execute.")

c = np.array(c)
s = np.array(s)

# -------------------------
# Plots finais (MF final, erro, saídas)
# -------------------------
# MFs finais (segunda linha dos subplots)
for j in range(nfp):
    for i in range(nin):
        w = gaussmf(xval, c[i, j], s[i, j])
        ax = axes[1, i] if nin > 1 else axes
        ax.plot(xval, w, color='black', linewidth=0.8)
        ax.set_title('Membership Functions - Final' if j == 0 else '')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)
fig_mf_init.tight_layout()

# Erro por geração
plt.figure()
plt.plot(erro, 'r', linewidth=1.5)
plt.xlabel('Geração')
plt.ylabel('EQM')
plt.title('Erro Quadrático Médio')
plt.grid(True)

# Saídas de treino/validação comparativas
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))
axs[0, 0].plot(y_train, 'r', label='Saída Desejada')
axs[0, 0].plot(ytst, 'k', label='Saída Inicial')
axs[0, 0].legend()
axs[0, 0].set_title('Treinamento - Saída Desejada x Saída Inicial')

axs[0, 1].plot(yst, 'g')
axs[0, 1].set_title('Treinamento - Saída Final')

axs[0, 2].plot(y_train, 'r', label='Saída Desejada')
axs[0, 2].plot(yst, 'g', label='Saída Final')
axs[0, 2].legend()
axs[0, 2].set_title('Treinamento - Saída Desejada x Saída Final')

axs[1, 0].plot(y_val, 'r', label='Saída Desejada')
axs[1, 0].plot(yvst, 'k', label='Saída Inicial')
axs[1, 0].legend()
axs[1, 0].set_title('Validação - Saída Desejada x Saída Inicial')

axs[1, 1].plot(ysv, 'g')
axs[1, 1].set_title('Validação - Saída Final')

axs[1, 2].plot(y_val, 'r', label='Saída Desejada')
axs[1, 2].plot(ysv, 'g', label='Saída Final')
axs[1, 2].legend()
axs[1, 2].set_title('Validação - Saída Desejada x Saída Final')

fig3.tight_layout()

# Figuras finais simples
plt.figure()
plt.plot(y_train, 'r', label='Saída Desejada')
plt.plot(yst, 'b', label='Saída GP-NFN-I')
plt.title('Treinamento - Saída Desejada x Saída Final')
plt.legend()

plt.figure()
plt.plot(y_val, 'r', label='Desired Output')
plt.plot(ysv, 'g', label='Estimated Output')
plt.title('Validation - Desired Output x Estimated Output')
plt.legend()

plt.show()