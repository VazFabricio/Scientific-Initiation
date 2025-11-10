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
ALFA = 0.01           
NEPOCA = 5

# Parâmetros AG
TAM_POP = 50
NUM_GERACOES = 5
TAXA_CRUZA = 0.9
TAXA_MUTA = 0.08
NFP_MAX = 5
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
# Partição: 60% treino / 40% validação (Teste desativado)
# -------------------------
npt_tr = int(round(npt_total * 0.6))

indices = np.arange(len(xt_all))
xt_all = xt_all[indices]
yt_all = yt_all[indices]

xt = xt_all[:npt_tr, :].copy()       # treino (60%)
ydt = yt_all[:npt_tr].copy()            # y de treino

xv = xt_all[npt_tr:, :].copy()  # validação (xv)
ydv = yt_all[npt_tr:].copy()    # y validação (ydv)

npt = npt_tr

# -------------------------
# limites das features
# -------------------------
xmin = xt.min(axis=0)
xmax = xt.max(axis=0)
delta = (xmax - xmin) / (NFP_INIT - 1)

# -------------------------
# Inicialização de p e q
# -------------------------
rng = np.random.default_rng()
p = rng.random((nin, NFP_INIT))
q = rng.random(NFP_INIT)

# -------------------------
# função gaussiana
# -------------------------
def gaussmf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))

# helper para extrair predição do retorno de saida.saida(...)
def extract_prediction(saida_ret):
    """
    saida.saida pode retornar:
      - array_like (predicoes)
      - tuple/list (ys, w, y_vec, b) ou (y_pred_array, ...)
    Aqui pegamos o primeiro elemento se é tupla/lista, senão usamos direto.
    """
    if isinstance(saida_ret, (list, tuple)):
        return np.asarray(saida_ret[0]).ravel()
    else:
        return np.asarray(saida_ret).ravel()

# helper para extrair escalar de objeto possivelmente 0-d ou array com 1 elemento
def scalar_of(x):
    arr = np.asarray(x).ravel()
    return float(arr[0])

# -------------------------
# Gera população inicial
# cada indivíduo terá chaves: 'nfps', 'cs', 'ss', 'saida', 'fitness'
# -------------------------
pop = []
for z in range(TAM_POP):
    nfpSort = NFP_INIT
    cs = np.empty((nin, nfpSort))
    ss = np.empty((nin, nfpSort))
    for j in range(nfpSort):
        for i in range(nin):
            cs[i, j] = xmin[i] + rng.random() * (xmax[i] - xmin[i])
            ss[i, j] = rng.random() * (xmax[i] - xmin[i])

    indiv = {'nfps': nfpSort, 'cs': cs, 'ss': ss}
    saida_full = saida.saida(xt, cs, ss, p, q, nfpSort)
    y_pred = extract_prediction(saida_full)
    indiv['saida'] = y_pred
    indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - ydt) ** 2)) / npt
    pop.append(indiv)

# identifica melhor indivíduo inicial (índice)
melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

# inicializa parâmetros com o melhor indivíduo
c = pop[melhorindv]['cs'].copy()
s = pop[melhorindv]['ss'].copy()
nfp = pop[melhorindv]['nfps']
novapop = pop.copy()

# -------------------------
# Preparar xval para plot das MF (apenas para primeira feature)
# -------------------------
xval = np.linspace(xmin[0], xmax[0], npt)

# configurar figura MF (2 linhas: inicial e final; colunas = nin)
fig_mf, axes = plt.subplots(2, max(1, nin), figsize=(4 * max(1, nin), 6))
axes = np.array(axes)
if axes.ndim == 1:
    axes = axes.reshape(2, 1)
elif axes.shape[0] != 2:
    axes = axes.reshape(2, nin)

# plot MFs iniciais
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
# Saída inicial sem treinamento
# -------------------------
yst = extract_prediction(saida.saida(xt, c, s, p, q, nfp))
ysv = extract_prediction(saida.saida(xv, c, s, p, q, nfp))

# -------------------------
# Treinamento (laço de gerações + atualização p,q por época)
# -------------------------
erro = []
y_train_pred = yst.copy()

for gen in range(NUM_GERACOES):
    print(f'Geração {gen+1}/{NUM_GERACOES}')
    erro.append((0.5 * np.sum((y_train_pred - ydt) ** 2)) / npt)
    dyjdqj = 1.0

    # atualização estilo gradiente para p e q
    for _ in range(NEPOCA):
        for k in range(npt):
            sample = xt[k, :]
            ys_full = saida.saida(sample, c, s, p, q, nfp)
            ys = scalar_of(ys_full[0])
            w = np.asarray(ys_full[1]).ravel()
            y_vec = np.asarray(ys_full[2]).ravel()
            b = scalar_of(ys_full[3])
            
            dedys = float(ys) - float(ydt[k])

            for j in range(nfp):
                dysdyj = w[j] / b
                for i in range(nin):
                    dyjdpj = sample[i]
                    p[i, j] = p[i, j] - ((ALFA / 10.0) * dedys * dysdyj * dyjdpj)
                q[j] = q[j] - ((ALFA / 10.0) * dedys * dysdyj * dyjdqj)

    # aplicar operador genético para gerar nova população
    pop = gerarnovapop.gerarnovapop(pop, melhorindv, TAM_POP, TAXA_CRUZA, TAXA_MUTA, xmax, xmin)

    # re-avaliar fitness da nova população (sempre usando os pesos p,q atuais e dados de treino)
    for z in range(TAM_POP):
        indiv = pop[z]
        saida_full = saida.saida(xt, indiv['cs'], indiv['ss'], p, q, indiv['nfps'])
        y_pred = extract_prediction(saida_full)
        indiv['saida'] = y_pred
        indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - ydt) ** 2)) / npt

    # atualizar melhor indivíduo
    melhorindv = int(np.argmin([ind['fitness'] for ind in pop]))

    # atualizar parâmetros a partir do melhor indivíduo
    c = pop[melhorindv]['cs'].copy()
    s = pop[melhorindv]['ss'].copy()
    nfp = pop[melhorindv]['nfps']

    # recomputar predição de treino com parâmetros atualizados
    y_train_pred = extract_prediction(saida.saida(xt, c, s, p, q, nfp))

# adiciona último erro
erro.append((0.5 * np.sum((y_train_pred - ydt) ** 2)) / npt)

# -------------------------
# Predições finais (com os parâmetros finais)
# -------------------------
y_train_pred_final = extract_prediction(saida.saida(xt, c, s, p, q, nfp))
y_val_pred_final = extract_prediction(saida.saida(xv, c, s, p, q, nfp))

end_time = time.time()
print(f"Tempo de execução: {end_time - start_time:.3f} s")

c = np.array(c)
s = np.array(s)

# -------------------------
# Métricas (validação e teste)
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
# Plots finais: MFs finais, erro por geração, comparativos treino/val
# -------------------------

# MFs finais
for j in range(nfp):
    for i in range(nin):
        w = gaussmf(xval, c[i, j], s[i, j])
        ax = axes[1, i]
        ax.plot(xval, w, linewidth=0.8)
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

# Plots comparativos (treino e validação)
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))

# Treino - desejada vs inicial vs final
axs[0, 0].plot(ydt, 'r', label='Saída Desejada (Treino)')
axs[0, 0].plot(yst, 'k', label='Saída Inicial (Treino)')
axs[0, 0].legend()
axs[0, 0].set_title('Treino - Desejada x Inicial')

axs[0, 1].plot(y_train_pred_final, label='Saída Final (Treino)')
axs[0, 1].set_title('Treino - Saída Final')

axs[0, 2].plot(ydt, 'r', label='Saída Desejada (Treino)')
axs[0, 2].plot(y_train_pred_final, 'g', label='Saída Final (Treino)')
axs[0, 2].legend()
axs[0, 2].set_title('Treino - Desejada x Final')

# Validação - desejada vs inicial vs final
axs[1, 0].plot(ydv, 'r', label='Saída Desejada (Val)')
axs[1, 0].plot(ysv, 'k', label='Saída Inicial (Val)')
axs[1, 0].legend()
axs[1, 0].set_title('Validação - Desejada x Inicial')

axs[1, 1].plot(y_val_pred_final, label='Saída Final (Val)')
axs[1, 1].set_title('Validação - Saída Final')

axs[1, 2].plot(ydv, 'r', label='Saída Desejada (Val)')
axs[1, 2].plot(y_val_pred_final, 'g', label='Saída Final (Val)')
axs[1, 2].legend()
axs[1, 2].set_title('Validação - Desejada x Final')

fig3.tight_layout()

# Figuras finais simples (treino, val)
plt.figure()
plt.plot(ydt, 'r', label='Desejada (Treino)')
plt.plot(y_train_pred_final, 'b', label='Estimada (Treino)')
plt.title('Treino - Desejada x Estimada')
plt.legend()

plt.figure()
plt.plot(ydv, 'r', label='Desejada (Validação)')
plt.plot(y_val_pred_final, 'g', label='Estimada (Validação)')
plt.title('Validação - Desejada x Estimada')
plt.legend()

# Scatter: Real vs Predito (Validação)
plt.figure(figsize=(6, 6))
plt.scatter(ydv, y_val_pred_final, alpha=0.6, label='Amostras (val)')
min_val = min(np.nanmin(ydv) if len(ydv) > 0 else 0, np.nanmin(y_val_pred_final) if len(y_val_pred_final) > 0 else 0)
max_val = max(np.nanmax(ydv) if len(ydv) > 0 else 1, np.nanmax(y_val_pred_final) if len(y_val_pred_final) > 0 else 1)
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='y = x')
plt.title('Dispersão: Saída Real vs Predita (Validação)')
plt.xlabel('Saída Real')
plt.ylabel('Saída Predita')
plt.legend()
plt.grid(True)

plt.show()