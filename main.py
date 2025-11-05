import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy import sparse
import saida
import gerarnovapop

start_time = time.time()

# -------------------------
# Parâmetros ajustáveis
# -------------------------
NUM_MF = 5            # Funções de pertinência iniciais por variável
learning_rate = 0.01  # taxa de aprendizado (alfa)
epochs_per_generation = 5  # número de épocas de atualização dos pesos por geração

# Parâmetros da Programação Genética
population_size = 50
num_generations = 50
crossover_rate = 0.9
mutation_rate = 0.08
max_mf_per_variable = 7  # máximo de FPs geradas por indivíduo (nfpMax)

# -------------------------
# Carregar dados (arquivos CSV com header)
# -------------------------
raw_X = np.loadtxt('xt.csv', delimiter=',', skiprows=1)
# assume que a primeira coluna era índice; pega colunas a partir de 1
features = raw_X[:, 1:]
raw_y = np.loadtxt('yt.csv', delimiter=',', skiprows=1)
targets = raw_y[:, 1:].ravel()   # ravel em caso de coluna única

# -------------------------
# Partição de treino/validação/teste (ajuste conforme necessidade)
# -------------------------
n_train = 2100
n_val = 700
n_test = 700

num_samples, num_features = features.shape

X_train = features[0:n_train, :]
y_train = targets[0:n_train]

X_val = features[n_train: n_train + n_val, :]
y_val = targets[n_train: n_train + n_val]

X_test = features[n_train + n_val: n_train + n_val + n_test, :]
y_test = targets[n_train + n_val: n_train + n_val + n_test]

# usado nos cálculos de fitness (npt no código original)
npt = n_train

# limites das features (baseados no treino)
feature_min = X_train.min(axis=0)
feature_max = X_train.max(axis=0)
delta = (feature_max - feature_min) / (NUM_MF - 1)

# -------------------------
# Inicialização de parâmetros p e q
# -------------------------
rng = np.random.default_rng()
p = rng.random((num_features, NUM_MF))
q = rng.random(NUM_MF)

# -------------------------
# Função gaussiana (MF)
# -------------------------
def gaussmf(x, mean, sigma):
    sigma = np.maximum(sigma, 1e-12)
    return np.exp(-((x - mean) ** 2) / (2.0 * sigma ** 2))

# -------------------------
# Helper para extrair predições do retorno de saida.saida(...)
# aceita tanto retorno direto (array) quanto tupla/lista cujo primeiro elemento é y_pred
# -------------------------
def extract_prediction(saida_ret):
    if isinstance(saida_ret, (list, tuple)):
        return np.asarray(saida_ret[0]).ravel()
    else:
        return np.asarray(saida_ret).ravel()

# -------------------------
# Gera população inicial
# cada indivíduo: dict com 'nfps', 'cs', 'ss', 'saida', 'fitness'
# -------------------------
population = []
for _ in range(population_size):
    nfps = NUM_MF
    cs = np.empty((num_features, nfps))
    ss = np.empty((num_features, nfps))
    for j in range(nfps):
        for i in range(num_features):
            cs[i, j] = feature_min[i] + rng.random() * (feature_max[i] - feature_min[i])
            ss[i, j] = rng.random() * (feature_max[i] - feature_min[i])  # sigma relativo ao range

    indiv = {'nfps': nfps, 'cs': cs, 'ss': ss}
    out = saida.saida(X_train, cs, ss, p, q, nfps)
    y_pred_train = extract_prediction(out)
    indiv['saida'] = y_pred_train
    indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - y_train) ** 2)) / npt
    population.append(indiv)

# identifica melhor indivíduo inicial
best_idx = int(np.argmin([ind['fitness'] for ind in population]))

# inicializar parâmetros com melhor indivíduo
best_cs = population[best_idx]['cs'].copy()
best_ss = population[best_idx]['ss'].copy()
current_nfps = population[best_idx]['nfps']
new_population = population.copy()

# -------------------------
# Preparar xval para plot das MF (apenas para a primeira feature como exemplo visual)
# -------------------------
xval = np.linspace(feature_min[0], feature_max[0], npt)

# plot - MF iniciais (2 linhas: inicial e final; colunas = num_features)
fig_mf, axes = plt.subplots(2, max(1, num_features), figsize=(4 * max(1, num_features), 6))
axes = np.array(axes)
if axes.ndim == 1:
    axes = axes.reshape(2, 1)
elif axes.shape[0] != 2:
    axes = axes.reshape(2, num_features)

for j in range(current_nfps):
    for i in range(num_features):
        w = gaussmf(xval, best_cs[i, j], best_ss[i, j])
        ax = axes[0, i]
        ax.plot(xval, w, linewidth=0.8)
        if j == 0:
            ax.set_title('Membership Functions - Inicial')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(feature_min[i], feature_max[i])
        ax.grid(True)

# -------------------------
# Saída inicial (antes de qualquer ajuste de p/q)
# -------------------------
y_train_pred_initial = extract_prediction(saida.saida(X_train, best_cs, best_ss, p, q, current_nfps))
y_val_pred_initial = extract_prediction(saida.saida(X_val, best_cs, best_ss, p, q, current_nfps))
y_test_pred_initial = extract_prediction(saida.saida(X_test, best_cs, best_ss, p, q, current_nfps))

# -------------------------
# Treinamento (laço de gerações + atualizações de p,q por época)
# -------------------------
error_history = []
y_train_pred = y_train_pred_initial.copy()

for gen in range(num_generations):
    print(f'Geração {gen+1}/{num_generations}')
    error_history.append((0.5 * np.sum((y_train_pred - y_train) ** 2)) / npt)
    dyjdqj = 1.0

    # treinamento dos pesos p e q (estilo gradiente)
    for _ in range(epochs_per_generation):
        for k in range(npt):
            sample = X_train[k, :]
            ys_full = saida.saida(sample, best_cs, best_ss, p, q, current_nfps)
            if isinstance(ys_full, (list, tuple)):
                ys = float(np.asarray(ys_full[0]).ravel())
                w = np.asarray(ys_full[1]).ravel()
                y_vec = np.asarray(ys_full[2])
                b = float(ys_full[3])
            else:
                raise RuntimeError("saida.saida não retornou (ys, w, y_vec, b) para amostra. Ajuste a função 'saida'.")

            dedys = ys - float(y_train[k])  # erro escalar para a amostra

            for j in range(current_nfps):
                dysdyj = w[j] / b
                for i in range(num_features):
                    dyjdpj = sample[i]
                    p[i, j] = p[i, j] - ((learning_rate / 10.0) * dedys * dysdyj * dyjdpj)
                q[j] = q[j] - ((learning_rate / 10.0) * dedys * dysdyj * dyjdqj)

    # gerar nova população via AG (usa função existente)
    population = gerarnovapop.gerarnovapop(population, best_idx, population_size, crossover_rate, mutation_rate, feature_max, feature_min)

    # re-avaliar fitness para cada indivíduo
    for z in range(population_size):
        indiv = population[z]
        out = saida.saida(X_train, indiv['cs'], indiv['ss'], p, q, indiv['nfps'])
        y_pred_vec = extract_prediction(out)
        indiv['saida'] = y_pred_vec
        indiv['fitness'] = (0.5 * np.sum((indiv['saida'] - y_train) ** 2)) / npt

    # atualizar melhor indivíduo
    best_idx = int(np.argmin([ind['fitness'] for ind in population]))

    # atualizar parâmetros com melhor indivíduo atual
    best_cs = population[best_idx]['cs'].copy()
    best_ss = population[best_idx]['ss'].copy()
    current_nfps = population[best_idx]['nfps']

    # recomputar predição de treino com os parâmetros atualizados
    y_train_pred = extract_prediction(saida.saida(X_train, best_cs, best_ss, p, q, current_nfps))

# erro final após última geração
error_history.append((0.5 * np.sum((y_train_pred - y_train) ** 2)) / npt)

# -------------------------
# Predições finais (treino, validação, teste) com parâmetros finais
# -------------------------
y_train_pred_final = extract_prediction(saida.saida(X_train, best_cs, best_ss, p, q, current_nfps))
y_val_pred_final = extract_prediction(saida.saida(X_val, best_cs, best_ss, p, q, current_nfps))
y_test_pred_final = extract_prediction(saida.saida(X_test, best_cs, best_ss, p, q, current_nfps))

end_time = time.time()
print(f"Tempo de execução: {end_time - start_time:.3f} segundos.")

best_cs = np.array(best_cs)
best_ss = np.array(best_ss)

# -------------------------
# Métricas de desempenho (usando predições finais no conjunto de teste)
# -------------------------
mse_test = mean_squared_error(y_test, y_test_pred_final)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred_final)

print("\n===== MÉTRICAS - CONJUNTO DE TESTE =====")
print(f"MSE  = {mse_test:.6f}")
print(f"RMSE = {rmse_test:.6f}")
print(f"R²   = {r2_test:.6f}")

# -------------------------
# Plots finais (MF final, erro, saídas)
# -------------------------

# MFs finais (segunda linha dos subplots)
for j in range(current_nfps):
    for i in range(num_features):
        w = gaussmf(xval, best_cs[i, j], best_ss[i, j])
        ax = axes[1, i]
        ax.plot(xval, w, linewidth=0.8)
        if j == 0:
            ax.set_title('Membership Functions - Final')
        ax.set_xlabel(f'X_{i+1}')
        ax.set_ylabel('Membership')
        ax.set_xlim(feature_min[i], feature_max[i])
        ax.grid(True)
fig_mf.tight_layout()

# Erro por geração
plt.figure()
plt.plot(error_history, linewidth=1.5)
plt.xlabel('Geração')
plt.ylabel('EQM')
plt.title('Erro Quadrático Médio por Geração')
plt.grid(True)

# Saídas de treino/validação comparativas (organização em subplots)
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))

# Treino — desejada vs inicial vs final
axs[0, 0].plot(y_train, 'r', label='Saída Desejada (Treino)')
axs[0, 0].plot(y_train_pred_initial, 'k', label='Saída Inicial (Treino)')
axs[0, 0].legend()
axs[0, 0].set_title('Treino - Desejada x Inicial')

axs[0, 1].plot(y_train_pred_final, label='Saída Final (Treino)')
axs[0, 1].set_title('Treino - Saída Final')

axs[0, 2].plot(y_train, 'r', label='Saída Desejada (Treino)')
axs[0, 2].plot(y_train_pred_final, 'g', label='Saída Final (Treino)')
axs[0, 2].legend()
axs[0, 2].set_title('Treino - Desejada x Final')

# Validação — desejada vs inicial vs final
axs[1, 0].plot(y_val, 'r', label='Saída Desejada (Val)')
axs[1, 0].plot(y_val_pred_initial, 'k', label='Saída Inicial (Val)')
axs[1, 0].legend()
axs[1, 0].set_title('Validação - Desejada x Inicial')

axs[1, 1].plot(y_val_pred_final, label='Saída Final (Val)')
axs[1, 1].set_title('Validação - Saída Final')

axs[1, 2].plot(y_val, 'r', label='Saída Desejada (Val)')
axs[1, 2].plot(y_val_pred_final, 'g', label='Saída Final (Val)')
axs[1, 2].legend()
axs[1, 2].set_title('Validação - Desejada x Final')

fig3.tight_layout()

# Figuras finais simples
plt.figure()
plt.plot(y_train, 'r', label='Saída Desejada (Treino)')
plt.plot(y_train_pred_final, 'b', label='Saída Estimada (Treino)')
plt.title('Treino - Desejada x Estimada')
plt.legend()

plt.figure()
plt.plot(y_val, 'r', label='Saída Desejada (Validação)')
plt.plot(y_val_pred_final, 'g', label='Saída Estimada (Validação)')
plt.title('Validação - Desejada x Estimada')
plt.legend()

plt.figure()
plt.plot(y_test, 'r', label='Saída Desejada (Teste)')
plt.plot(y_test_pred_final, 'g', label='Saída Estimada (Teste)')
plt.title('Teste - Desejada x Estimada')
plt.legend()

# -------------------------
# Gráfico: Saída Real vs Predita (Teste)
# -------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred_final, alpha=0.6, label='Amostras')
min_val = min(y_test.min(), y_test_pred_final.min())
max_val = max(y_test.max(), y_test_pred_final.max())
plt.plot([min_val, max_val], [min_val, max_val], '--', linewidth=1.5, label='y = x')
plt.title('Dispersão: Saída Real vs. Predita (Teste)')
plt.xlabel('Saída Real')
plt.ylabel('Saída Predita')
plt.legend()
plt.grid(True)

plt.show()
