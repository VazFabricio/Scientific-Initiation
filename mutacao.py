import numpy as np
from mapa_caotico import ChaoticRNG

_caos_mut = ChaoticRNG(x0=0.54321, r=3.99)

def mutacao(indv, xmax, xmin):
    nin = len(indv['cs'])
    nfp = len(indv['cs'][0])

    i = round(1 + np.random.rand() * (nin - 1)) - 1
    j = round(1 + np.random.rand() * (nfp - 1)) - 1
    # indv['cs'][i][j] = np.random.rand() * (xmax[i] - xmin[i])
    # indv['ss'][i][j] = np.random.rand() * (xmax[i] - xmin[i])
    indv['cs'][i][j] = (_caos_mut.next() *   (xmax[i] - xmin[i]))
    indv['ss'][i][j] = (_caos_mut.next() * (xmax[i] - xmin[i]))

    #Mutação do Consequente (p e q)
    i_cons = round(np.random.rand() * (nin - 1))
    j_cons = round(np.random.rand() * (nfp - 1))

    # indv['p'][i_cons][j_cons] = np.random.rand()
    # indv['q'][j_cons] = np.random.rand()
    indv['p'][i_cons][j_cons] = _caos_mut.next()
    indv['q'][j_cons] = _caos_mut.next()
    
    return indv
