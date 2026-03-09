import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# ====================================================
USAR_NORMALIZACAO = False
# ====================================================

resultados_rmse = []

for exec in range(21):
    print(f'exec: {exec}')
    
    # -------------------------
    # Parâmetros ajustáveis
    # -------------------------
    NFP_INIT = 5        
    ALFA = 0.01          
    NEPOCA = 100         
    FILE_XT = 'xt_inflow.csv'
    FILE_YT = 'yt_inflow.csv'

    start_time = time.time()

    # -------------------------
    # Carregar Dados
    # -------------------------
    _raw_X = np.loadtxt(FILE_XT, delimiter=',', skiprows=1)
    xt_all = _raw_X[:, 1:]
    _raw_y = np.loadtxt(FILE_YT, delimiter=',', skiprows=1)
    yt_all = _raw_y[:, 1:].ravel()

    npt_total, nin = xt_all.shape
    npt_tr = int(round(npt_total * 0.6))

    # Limites reais para cálculo dos centros das Gaussianas
    x_min_real = xt_all[:npt_tr, :].min(axis=0)
    x_max_real = xt_all[:npt_tr, :].max(axis=0)

    # -------------------------
    # Preparação dos Dados (Com ou Sem Normalização)
    # -------------------------
    if USAR_NORMALIZACAO:
        xt = (xt_all[:npt_tr, :] - x_min_real) / (x_max_real - x_min_real + 1e-10)
        xv = (xt_all[npt_tr:, :] - x_min_real) / (x_max_real - x_min_real + 1e-10)
        
        y_min = yt_all[:npt_tr].min()
        y_max = yt_all[:npt_tr].max()
        ydt = (yt_all[:npt_tr] - y_min) / (y_max - y_min + 1e-10)
        
        # Se normalizou, os centros vão de 0 a 1 para todas as entradas
        c_min = np.zeros(nin)
        c_max = np.ones(nin)
    else:
        xt = xt_all[:npt_tr, :].copy()
        xv = xt_all[npt_tr:, :].copy()
        ydt = yt_all[:npt_tr].copy()
        
        # Se não normalizou, os centros acompanham os limites reais de cada entrada
        c_min = x_min_real
        c_max = x_max_real

    ydv_real = yt_all[npt_tr:] # Gabarito real para validação (nunca muda)

    # -------------------------
    # Inicialização do NFN
    # -------------------------
    centros = np.zeros((nin, NFP_INIT))
    sigmas = np.zeros((nin, 1))

    # Distribuindo as Gaussianas perfeitamente para cada variável
    for i in range(nin):
        centros[i, :] = np.linspace(c_min[i], c_max[i], NFP_INIT)
        sigmas[i, 0] = (c_max[i] - c_min[i]) / (NFP_INIT - 1) + 1e-10

    # Pesos Consequentes (W) inicializados com valores pequenos
    W = np.random.uniform(-0.1, 0.1, (nin, NFP_INIT))

    def fuzificacao(x_amostra):
        """Calcula o grau de pertinência (mu) para uma amostra. Adaptativo ao range da variável."""
        return np.exp(-((x_amostra[:, None] - centros) ** 2) / (2 * sigmas ** 2))

    # -------------------------
    # Treinamento: Gradiente Descendente do NFN
    # -------------------------
    for epoca in range(NEPOCA):
        indices = np.random.permutation(npt_tr)
        for idx in indices:
            x_k = xt[idx]
            y_real = ydt[idx]

            # 1. Forward
            mu = fuzificacao(x_k)
            y_pred = np.sum(mu * W)

            # 2. Erro
            erro_k = y_pred - y_real

            # 3. Backward
            W = W - ALFA * erro_k * mu

    # -------------------------
    # Predição e Métricas
    # -------------------------
    def predict(X):
        y_out = []
        for x_val in X:
            mu = fuzificacao(x_val)
            y_out.append(np.sum(mu * W))
        return np.array(y_out)

    y_val_pred_raw = predict(xv)
    
    if USAR_NORMALIZACAO:
        y_val_pred = y_val_pred_raw * (y_max - y_min + 1e-10) + y_min
    else:
        y_val_pred = y_val_pred_raw

    rmse_val = np.sqrt(mean_squared_error(ydv_real, y_val_pred))
    r2_val = r2_score(ydv_real, y_val_pred)

    print(f"Validação -> RMSE: {rmse_val:.6f}  R2: {r2_val:.6f}".replace('.', ','))
    resultados_rmse.append(f"{rmse_val:.6f}".replace('.', ','))

print("\n--- COPIE PARA A PLANILHA ---")
print("\n".join(resultados_rmse))