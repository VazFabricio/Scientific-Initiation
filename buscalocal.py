"""
Implementar um refinamento estocástico (Hill Climbing) para ajustar centros e larguras das funções de pertinência Gaussianas.
"""
import numpy as np
import saida

def busca_local_antecedentes(indv, xt, ydt, p, q, nfp, step_size=0.01):
    
    def calc_erro(ind):
        ret = saida.saida(xt, ind['cs'], ind['ss'], p, q, nfp)
        y_pred = ret[0] if isinstance(ret, (list, tuple)) else ret
        y_pred = np.asarray(y_pred).ravel()
        return (0.5 * np.sum((y_pred - ydt) ** 2)) / len(ydt)

    melhor_erro = calc_erro(indv)
    nin = len(indv['cs'])
    
    # Itera sobre os parametros para tentar refinamento
    for j in range(nfp):
        for i in range(nin):
            # --- Refino do Centro (c) ---
            original_c = indv['cs'][i][j]
            
            indv['cs'][i][j] = original_c + step_size
            erro_pos = calc_erro(indv)
            
            if erro_pos < melhor_erro:
                melhor_erro = erro_pos
            else:
                indv['cs'][i][j] = original_c - step_size
                erro_neg = calc_erro(indv)
                if erro_neg < melhor_erro:
                    melhor_erro = erro_neg
                else:
                    indv['cs'][i][j] = original_c

            # --- Refino da Largura (s) ---
            original_s = indv['ss'][i][j]
            
            indv['ss'][i][j] = original_s + step_size
            erro_pos = calc_erro(indv)
            
            if erro_pos < melhor_erro:
                melhor_erro = erro_pos
            else:
                indv['ss'][i][j] = original_s - step_size
                if indv['ss'][i][j] < 1e-4: 
                    indv['ss'][i][j] = 1e-4
                
                erro_neg = calc_erro(indv)
                if erro_neg < melhor_erro:
                    melhor_erro = erro_neg
                else:
                    indv['ss'][i][j] = original_s

    indv['fitness'] = melhor_erro
    return indv