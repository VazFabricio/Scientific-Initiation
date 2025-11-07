import numpy as np
import copy

def mutacao(indv, xmax, xmin):
    # Crie uma cópia profunda para não modificar o original
    novo_indv = copy.deepcopy(indv) 
    
    nin = len(novo_indv['cs'])
    nfp = len(novo_indv['cs'][0])

    i = round(1 + np.random.rand() * (nin - 1)) - 1
    j = round(1 + np.random.rand() * (nfp - 1)) - 1

    novo_indv['cs'][i][j] = np.random.rand() * (xmax[i] - xmin[i])
    novo_indv['ss'][i][j] = np.random.rand() * (xmax[i] - xmin[i])
    
    return novo_indv 