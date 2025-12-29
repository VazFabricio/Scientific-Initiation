"""
Implementar um refinamento estocástico (Hill Climbing) para ajustar centros e larguras das funções de pertinência Gaussianas.
"""
import numpy as np
import saida


def busca_local_estocastica(indv, xt, ydt, p, q, nfp, step_size=0.01):
    """
    Realiza uma busca local estocástica aplicando perturbações gaussianas nos parâmetros.

    Args:
        indv (dict): Estrutura do indivíduo com parâmetros gaussianos.
        xt (np.array): Entradas do conjunto de dados.
        ydt (np.array): Saídas esperadas (target).
        p, q (int): Parâmetros de configuração do modelo.
        nfp (int): Número de funções de pertinência por entrada.
        step_size (float): Desvio padrão da perturbação gaussiana aplicada.

    Returns:
        dict: Indivíduo com parâmetros potencialmente melhorados.
    """

    def calc_erro(ind):
        ret = saida.saida(xt, ind['cs'], ind['ss'], p, q, nfp)
        y_pred = ret[0] if isinstance(ret, (list, tuple)) else ret
        y_pred = np.asarray(y_pred).ravel()
        return (0.5 * np.sum((y_pred - ydt) ** 2)) / len(ydt)

    melhor_erro = calc_erro(indv)
    nin = len(indv['cs'])
    
    attempts_per_param = 3 

    for j in range(nfp):
        for i in range(nin):
            original_c = indv['cs'][i][j]
            
            melhorou_c = False
            for _ in range(attempts_per_param):
                noise = np.random.normal(0, step_size)
                indv['cs'][i][j] = original_c + noise
                erro_temp = calc_erro(indv)
                
                if erro_temp < melhor_erro:
                    melhor_erro = erro_temp
                    original_c = indv['cs'][i][j] 
                    melhorou_c = True
                else:
                    indv['cs'][i][j] = original_c 
            
            if not melhorou_c:
                indv['cs'][i][j] = original_c

            original_s = indv['ss'][i][j]
            melhorou_s = False
            
            for _ in range(attempts_per_param):
                noise = np.random.normal(0, step_size)
                novo_s = original_s + noise
                if novo_s < 1e-4: 
                    novo_s = 1e-4
                
                indv['ss'][i][j] = novo_s
                erro_temp = calc_erro(indv)
                
                if erro_temp < melhor_erro:
                    melhor_erro = erro_temp
                    original_s = indv['ss'][i][j]
                    melhorou_s = True
                else:
                    indv['ss'][i][j] = original_s
            
            if not melhorou_s:
                indv['ss'][i][j] = original_s

    indv['fitness'] = melhor_erro
    return indv