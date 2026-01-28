import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import saida
import gerarnovapop

for exec in range(21):
    print("exec: ", exec)
    # -------------------------
    # Parâmetros ajustáveis
    # -------------------------
    NFP_INIT = 5
    ALFA = 0.01           
    NEPOCA = 5

    # Parâmetros AG
    TAM_POP = 50
    NUM_GERACOES = 300
    TAXA_CRUZA = 0.9
    TAXA_MUTA = 0.08
    NFP_MAX = 5
    FILE_XT = 'xt_ccp.csv'
    FILE_YT = 'yt_ccp.csv'

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

    npt = npt_tr

    # 2. Normalização das Entradas (X)
    # Calculamos o min/max apenas no treino para evitar vazamento de dados
    x_min_train = xt_all[:npt_tr, :].min(axis=0)
    x_max_train = xt_all[:npt_tr, :].max(axis=0)
    xt_all_norm = (xt_all - x_min_train) / (x_max_train - x_min_train + 1e-10)

    # 3. Normalização das Saídas (y)
    y_min_train = yt_all[:npt_tr].min()
    y_max_train = yt_all[:npt_tr].max()
    yt_all_norm = (yt_all - y_min_train) / (y_max_train - y_min_train + 1e-10)

    # 4. Atribuição das variáveis de trabalho
    xt = xt_all_norm[:npt_tr, :].copy() 
    ydt = yt_all_norm[:npt_tr].copy()

    xv = xt_all_norm[npt_tr:, :].copy()
    ydv_norm = yt_all_norm[npt_tr:].copy() # Versão normalizada para validação
    ydv = yt_all[npt_tr:].copy()           # Versão original para métricas finais

    # Limites para o AG (agora sempre entre 0 e 1)
    xmin = np.zeros(nin)
    xmax = np.ones(nin)

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
        if gen % 50 == 0:
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
    c = np.array(c)
    s = np.array(s)
    """
    Módulo de Pós-processamento e Avaliação de Resultados.

    Este bloco realiza a desnormalização das predições para a escala original,
    calcula métricas estatísticas de erro e gera gráficos comparativos entre
    os dados reais e as estimativas do modelo neuro-fuzzy.
    """

    # -------------------------
    # Predições finais (Normalizadas)
    # -------------------------
    y_train_norm = extract_prediction(saida.saida(xt, c, s, p, q, nfp))
    y_val_norm = extract_prediction(saida.saida(xv, c, s, p, q, nfp))
    y_init_train_norm = yst # Saída inicial salva antes do treino
    y_init_val_norm = ysv     # Saída inicial salva antes do treino

    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time:.3f} s")

    # -------------------------
    # Função de Desnormalização
    # -------------------------
    def denorm(y_scaled):
        return y_scaled * (y_max_train - y_min_train + 1e-10) + y_min_train

    # Conversão para escala original
    y_train_pred_final = denorm(y_train_norm)
    y_val_pred_final = denorm(y_val_norm)
    yst_real = denorm(y_init_train_norm)
    ysv_real = denorm(y_init_val_norm)
    ydt_real = denorm(ydt) 
    # ydv já está na escala real conforme definido no bloco de carregamento

    # -------------------------
    # Métricas (Escala Real)
    # -------------------------
    mse_train = 0.5 * mean_squared_error(ydt_real, y_train_pred_final)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(ydt_real, y_train_pred_final)

    mse_val = 0.5 * mean_squared_error(ydv, y_val_pred_final)
    rmse_val = np.sqrt(mse_val)
    r2_val = r2_score(ydv, y_val_pred_final)

    print("\n===== MÉTRICAS (VALORES REAIS) =====")
    print(f"Treino   -> RMSE: {rmse_train:.6f}  R2: {r2_train:.6f}")
    print(f"Validação-> RMSE: {rmse_val:.6f}  R2: {r2_val:.6f}")
