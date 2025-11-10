import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import saida
import gerarnovapop

# -------------------------
# Parâmetros ajustáveis
# -------------------------
NFP_INIT = 5           
TAM_POP = 50
NUM_GERACOES = 300
TAXA_CRUZA = 0.9
TAXA_MUTA = 0.08
NFP_MAX = 7

FILE_XT = 'xt.csv'
FILE_YT = 'yt.csv'

start_time = time.time()

# -------------------------
# Carregar dados
# -------------------------
_raw_X = np.loadtxt(FILE_XT, delimiter=',', skiprows=1)
xt_all = _raw_X[:, 1:]        
_raw_y = np.loadtxt(FILE_YT, delimiter=',', skiprows=1)
yt_all = _raw_y[:, 1:].ravel()  

npt_total, nin = xt_all.shape

# -------------------------
# Partição: 60% treino / 40% validação
# -------------------------
npt_tr = int(round(npt_total * 0.6))


xt = xt_all[:npt_tr, :].copy()           
ydt = yt_all[:npt_tr].copy()             
xv = xt_all[npt_tr:, :].copy()  # validação (xv)
ydv = yt_all[npt_tr:].copy()    # y validação (ydv)   

npt = npt_tr  

# -------------------------
# Limites das features
# -------------------------
xmin = xt.min(axis=0)
xmax = xt.max(axis=0)
delta = (xmax - xmin) / (NFP_INIT - 1)

# -------------------------
# função gaussiana (para plot das MFs)
# -------------------------
def gaussmf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))

# -------------------------
# Gera população inicial
# -------------------------
rng = np.random.default_rng()
pop = []
for z in range(TAM_POP):
    nfpSort = NFP_INIT
    cs = np.empty((nin, nfpSort))
    ss = np.empty((nin, nfpSort))
    p_ind = rng.random((nin, nfpSort))
    q_ind = rng.random(nfpSort)

    for j in range(nfpSort):
        for i in range(nin):
            cs[i, j] = xmin[i] + rng.random() * (xmax[i] - xmin[i])
            ss[i, j] = rng.random() * (xmax[i] - xmin[i])

    indiv = {'nfps': nfpSort, 'cs': cs, 'ss': ss, 'p': p_ind, 'q': q_ind}
    saida_full = saida.saida(xt, cs, ss, p_ind, q_ind, nfpSort)
    y_pred = np.asarray(saida_full[0] if isinstance(saida_full, (list, tuple)) else saida_full).ravel()
    indiv['saida'] = y_pred
    indiv['fitness'] = (0.5 * np.sum((y_pred - ydt) ** 2)) / npt
    pop.append(indiv)

melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

c = pop[melhorindv]['cs'].copy()
s = pop[melhorindv]['ss'].copy()
nfp = pop[melhorindv]['nfps']
p = pop[melhorindv]['p']
q = pop[melhorindv]['q']

# -------------------------
# Preparar xval para plot das MF (visto que usaremos 2 linhas: inicial e final)
# -------------------------
xval = np.linspace(xmin[0], xmax[0], max(200, npt))

# configurar figura MF (2 linhas: inicial e final; colunas = nin)
fig_mf, axes = plt.subplots(2, max(1, nin), figsize=(4 * max(1, nin), 6))
axes = np.array(axes)
if axes.ndim == 1:
    axes = axes.reshape(2, 1)
elif axes.shape[0] != 2:
    axes = axes.reshape(2, nin)

# plot MFs iniciais (usando c,s do melhor indivíduo inicial)
for j in range(nfp):
    for i in range(nin):
        w = gaussmf(xval, c[i, j], s[i, j])
        ax = axes[0, i]
        ax.plot(xval, w, linewidth=0.8)
        if j == 0:
            ax.set_title('Membership Functions - Inicial')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)

# -------------------------
# Saídas iniciais (sem treinamento)
# -------------------------
yst = np.asarray(saida.saida(xt, c, s, p, q, nfp)[0]).ravel()
ysv = np.asarray(saida.saida(xv, c, s, p, q, nfp)[0]).ravel()

# -------------------------
# Loop de gerações
# -------------------------
erro = []
for gen in range(NUM_GERACOES):
    print(f'Geração {gen+1}/{NUM_GERACOES}')
    erro.append((0.5 * np.sum((yst - ydt) ** 2)) / npt)

    pop = gerarnovapop.gerarnovapop(pop, melhorindv, TAM_POP, TAXA_CRUZA, TAXA_MUTA, xmax, xmin)

    for z in range(TAM_POP):
        indiv = pop[z]
        saida_full = saida.saida(xt, indiv['cs'], indiv['ss'], indiv['p'], indiv['q'], indiv['nfps'])
        y_pred = np.asarray(saida_full[0] if isinstance(saida_full, (list, tuple)) else saida_full).ravel()
        indiv['saida'] = y_pred
        indiv['fitness'] = (0.5 * np.sum((y_pred - ydt) ** 2)) / npt

    melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

    c = pop[melhorindv]['cs'].copy()
    s = pop[melhorindv]['ss'].copy()
    nfp = pop[melhorindv]['nfps']
    p = pop[melhorindv]['p']
    q = pop[melhorindv]['q']

    yst = np.asarray(saida.saida(xt, c, s, p, q, nfp)[0]).ravel()

erro.append((0.5 * np.sum((yst - ydt) ** 2)) / npt)

# -------------------------
# Predições finais
# -------------------------
y_train_pred_final = np.asarray(saida.saida(xt, c, s, p, q, nfp)[0]).ravel()
y_val_pred_final = np.asarray(saida.saida(xv, c, s, p, q, nfp)[0]).ravel()

end_time = time.time()
print(f"\nTempo de execução: {end_time - start_time:.3f} s")

# -------------------------
# Métricas
# -------------------------
mse_val = 0.5 * mean_squared_error(ydv, y_val_pred_final)
rmse_val = np.sqrt(mse_val)
r2_val = r2_score(ydv, y_val_pred_final)

mse_train = 0.5 * mean_squared_error(ydt, y_train_pred_final)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(ydt, y_train_pred_final)

print("\n===== MÉTRICAS =====")
print(f"Treino   -> RMSE: {rmse_train:.6f}  R2: {r2_train:.6f}")
print(f"Validação-> RMSE: {rmse_val:.6f}  R2: {r2_val:.6f}")

# -------------------------
# Plots
# -------------------------
# Plot MFs finais (segunda linha dos subplots)
c = np.array(c)
s = np.array(s)
for j in range(nfp):
    for i in range(nin):
        w_final = gaussmf(xval, c[i, j], s[i, j])
        ax = axes[1, i]
        ax.plot(xval, w_final, linewidth=0.8)
        if j == 0:
            ax.set_title('Membership Functions - Final')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(xmin[i], xmax[i])
        ax.grid(True)
fig_mf.tight_layout()

# Erro por geração
plt.figure()
plt.plot(erro, 'r', linewidth=1.5)
plt.xlabel('Geração')
plt.ylabel('EQM (treino)')
plt.title('Erro Quadrático Médio por Geração (treino)')
plt.grid(True)

# Comparativos
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))

# Treino
axs[0, 0].plot(ydt, 'r', label='Desejada (Treino)')
axs[0, 0].plot(yst, 'k', label='Inicial (Treino)')
axs[0, 0].legend()
axs[0, 0].set_title('Treino - Desejada x Inicial')

axs[0, 1].plot(y_train_pred_final, label='Final (Treino)')
axs[0, 1].set_title('Treino - Saída Final')

axs[0, 2].plot(ydt, 'r', label='Desejada')
axs[0, 2].plot(y_train_pred_final, 'g', label='Final')
axs[0, 2].legend()
axs[0, 2].set_title('Treino - Desejada x Final')

# Validação
axs[1, 0].plot(ydv, 'r', label='Desejada (Val)')
axs[1, 0].plot(ysv, 'k', label='Inicial (Val)')
axs[1, 0].legend()
axs[1, 0].set_title('Validação - Desejada x Inicial')

axs[1, 1].plot(y_val_pred_final, label='Final (Val)')
axs[1, 1].set_title('Validação - Saída Final')

axs[1, 2].plot(ydv, 'r', label='Desejada')
axs[1, 2].plot(y_val_pred_final, 'g', label='Final')
axs[1, 2].legend()
axs[1, 2].set_title('Validação - Desejada x Final')

fig3.tight_layout()

# Scatter Real vs Predito (Validação)
plt.figure(figsize=(6, 6))
plt.scatter(ydv, y_val_pred_final, alpha=0.6, label='Amostras (val)')
min_val = min(np.min(ydv), np.min(y_val_pred_final))
max_val = max(np.max(ydv), np.max(y_val_pred_final))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y = x')
plt.title('Dispersão: Saída Real vs Predita (Validação)')
plt.xlabel('Saída Real')
plt.ylabel('Saída Predita')
plt.legend()
plt.grid(True)

plt.show()
