import numpy as np
import copy 

def mutacao(indv, xmax, xmin):

    novo_indv = copy.deepcopy(indv)
    
    nin = len(novo_indv['cs'])
    nfp = len(novo_indv['cs'][0])

    i = round(1 + np.random.rand() * (nin - 1)) - 1
    j = round(1 + np.random.rand() * (nfp - 1)) - 1

    novo_indv['cs'][i][j] = np.random.rand() * (xmax[i] - xmin[i])
    novo_indv['ss'][i][j] = np.random.rand() * (xmax[i] - xmin[i])

    #Mutação do Consequente (p e q)
    i_cons = round(np.random.rand() * (nin - 1))
    j_cons = round(np.random.rand() * (nfp - 1))

    novo_indv['p'][i_cons][j_cons] = np.random.rand()
    novo_indv['q'][j_cons] = np.random.rand()
    
    return novo_indv