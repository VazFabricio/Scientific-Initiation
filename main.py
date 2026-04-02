import time
import numpy as np
from anfis_toolbox import ANFISRegressor
from sklearn.metrics import mean_squared_error, r2_score

USAR_NORMALIZACAO = True

for exec in range(1):
    # -------------------------
    # Parâmetros ajustáveis
    # -------------------------
    ALFA = 0.01
    MAX_ITER = 300
    MFS = 5
    FILE_XT = 'xt_CN.csv'
    FILE_YT = 'yt_CN_u.csv'

    # -------------------------
    # Carregar dados
    # -------------------------
    _raw_X = np.loadtxt(FILE_XT, delimiter=',', skiprows=1)
    xt_all = _raw_X[:, 1:]
    _raw_y = np.loadtxt(FILE_YT, delimiter=',', skiprows=1)
    yt_all = _raw_y[:, 1:].ravel()

    xt_all = xt_all.astype(np.float32)
    yt_all = yt_all.astype(np.float32)

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
    # Instanciação e Treinamento do anfis
    # -------------------------
    anfis = ANFISRegressor(
        n_mfs=MFS,
        mf_type="gaussian",
        optimizer="hybrid_adam",
        learning_rate=ALFA,
        epochs=MAX_ITER,
        verbose=True
    )

    anfis.fit(xt, ydt)

    # -------------------------
    # Predições
    # -------------------------
    y_train_pred = anfis.predict(xt)
    y_val_pred = anfis.predict(xv)

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

    # -------------------------
    # Métricas (Escala Real)
    # -------------------------
    from sklearn.metrics import mean_squared_error

    mse_train = 0.5 * mean_squared_error(ydt_real, y_train_pred_final)
    rmse_train = np.sqrt(mse_train)

    mse_val = 0.5 * mean_squared_error(ydv_real, y_val_pred_final)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(ydv_real, y_val_pred_final)

    texto = f"Validação-> RMSE: {rmse_val:.6f}  R2: {r2_val:.6f}"
    print(texto.replace('.', ','))
    print("-" * 40)