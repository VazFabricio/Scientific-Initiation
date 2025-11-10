import numpy as np
import skfuzzy as fuzz

def saida(x_orig, c, s, p, q, m):
    """
    Versão vetorizada e rápida da função 'saida' original.
    Mantém a equivalência numérica com a implementação com loops.
    """
    # --- Conversão para arrays ---
    x = np.asarray(x_orig, dtype=float)
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # --- Garantir formato consistente ---
    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim != 2:
        raise ValueError("x_orig deve ser escalar, array 1D ou 2D.")

    n_amostras, n = x.shape
    n_regras = m

    # --- Expandir dimensões para cálculo vetorizado ---
    # x_exp: (n_amostras, n, 1)
    # c_exp, s_exp: (1, n, n_regras)
    x_exp = x[:, :, np.newaxis]
    c_exp = c[np.newaxis, :, :]
    s_exp = s[np.newaxis, :, :]

    # --- Cálculo das pertinências gaussianas ---
    # Usa a mesma fórmula de fuzz.gaussmf, vetorizada
    s_safe = np.where(s_exp == 0, 1e-9, s_exp)
    w_matriz = np.exp(-((x_exp - c_exp) ** 2) / (2 * s_safe ** 2))  # (n_amostras, n, m)

    # --- Produto das pertinências (regra) ---
    w_final = np.prod(w_matriz, axis=1)  # (n_amostras, m)

    # --- Cálculo da saída linear de cada regra ---
    # y_j = q_j + sum_k(p_kj * x_k)
    y_final = x @ p + q  # (n_amostras, m)

    # --- Agregação ponderada ---
    a = np.sum(w_final * y_final, axis=1)
    b = np.sum(w_final, axis=1)
    b = np.where(b == 0, 1.0, b)

    ys = a / b

    return ys, w_final, y_final, b
