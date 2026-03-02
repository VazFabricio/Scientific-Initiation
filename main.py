import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# ====================================================
# CHAVE DE CONFIGURAÇÃO
# ====================================================
USAR_NORMALIZACAO = False
# ====================================================

resultados_rmse = []

for exec in range(21):
    # -------------------------
    # Parâmetros ajustáveis
    # -------------------------
    ALFA = 0.01    
    MAX_ITER = 500       
    FILE_XT = 'xt_MG.csv'
    FILE_YT = 'yt_MG.csv'

    start_time = time.time()

    # -------------------------
    # Carregar dados
    # -------------------------
    _raw_X = np.loadtxt(FILE_XT, delimiter=',', skiprows=1)
    xt_all = _raw_X[:, 1:]
    _raw_y = np.loadtxt(FILE_YT, delimiter=',', skiprows=1)
    yt_all = _raw_y[:, 1:].ravel()

    npt_total, nin = xt_all.shape
    npt_tr = int(round(npt_total * 0.6))

    # -------------------------
    # Preparação dos Dados (Com ou Sem Normalização)
    # -------------------------
    if USAR_NORMALIZACAO:
        # Normaliza Entradas
        x_min_train = xt_all[:npt_tr, :].min(axis=0)
        x_max_train = xt_all[:npt_tr, :].max(axis=0)
        xt_all_norm = (xt_all - x_min_train) / (x_max_train - x_min_train + 1e-10)

        # Normaliza Saídas
        y_min_train = yt_all[:npt_tr].min()
        y_max_train = yt_all[:npt_tr].max()
        yt_all_norm = (yt_all - y_min_train) / (y_max_train - y_min_train + 1e-10)

        # Atribuição normalizada
        xt = xt_all_norm[:npt_tr, :].copy() 
        ydt = yt_all_norm[:npt_tr].copy()
        xv = xt_all_norm[npt_tr:, :].copy()
    else:
        # Atribuição bruta (sem normalizar)
        xt = xt_all[:npt_tr, :].copy()    
        ydt = yt_all[:npt_tr].copy()     
        xv = xt_all[npt_tr:, :].copy()  

    # Variáveis reais para usar nas métricas finais (nunca mudam)
    ydt_real = yt_all[:npt_tr].copy()
    ydv_real = yt_all[npt_tr:].copy()

    # -------------------------
    # Instanciação e Treinamento do MLP
    # -------------------------
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        activation='relu',          
        solver='sgd',               
        learning_rate_init=ALFA,    
        max_iter=MAX_ITER,          
        random_state=None           
    )

    mlp.fit(xt, ydt)

    # -------------------------
    # Predições
    # -------------------------
    y_train_pred = mlp.predict(xt)
    y_val_pred = mlp.predict(xv)

    # -------------------------
    # Desnormalização (se ativada)
    # -------------------------
    if USAR_NORMALIZACAO:
        def denorm(y_scaled):
            return y_scaled * (y_max_train - y_min_train + 1e-10) + y_min_train

        y_train_pred_final = denorm(y_train_pred)
        y_val_pred_final = denorm(y_val_pred)
    else:
        # Se não normalizou, as predições já estão na escala final
        y_train_pred_final = y_train_pred
        y_val_pred_final = y_val_pred

    end_time = time.time()

    # -------------------------
    # Métricas (Escala Real)
    # -------------------------
    mse_train = 0.5 * mean_squared_error(ydt_real, y_train_pred_final)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(ydt_real, y_train_pred_final)

    mse_val = 0.5 * mean_squared_error(ydv_real, y_val_pred_final)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(ydv_real, y_val_pred_final)

    texto = f"Validação-> RMSE: {rmse_val:.6f}  R2: {r2_val:.6f}"
    print(texto.replace('.', ','))
    print("-" * 40)
    
    valor_formatado = f"{rmse_val:.6f}".replace('.', ',')
    resultados_rmse.append(valor_formatado)
    
print("\n--- COPIE A LINHA ABAIXO E COLE NA PRIMEIRA CÉLULA VAZIA DA PLANILHA ---")
print("\n".join(resultados_rmse))