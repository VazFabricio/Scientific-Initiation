import numpy as np
import copy

# -------------------------
# Configurações
# -------------------------
MUTATION_TYPE = "tlbm"   # "classic" ou "tlbm"
ALPHA = 0.5
SIGMA = 0.05

# -------------------------
# Mutação clássica (100% compatível)
# -------------------------
def classic_mutation(indv, xmax, xmin):
    novo = copy.deepcopy(indv)

    nin = len(novo['cs'])
    nfp = len(novo['cs'][0])

    i = np.random.randint(nin)
    j = np.random.randint(nfp)

    novo['cs'][i][j] = xmin[i] + np.random.rand() * (xmax[i] - xmin[i])
    novo['ss'][i][j] = np.random.rand() * (xmax[i] - xmin[i])

    return novo

# -------------------------
# Teaching–Learning-Based Mutation
# -------------------------
def tlbm_mutation(indv, teacher, xmax, xmin):
    novo = copy.deepcopy(indv)
    TF = np.random.rand()

    nin = len(novo['cs'])
    nfp = len(novo['cs'][0])

    for i in range(nin):
        for j in range(nfp):
            # Atualiza centros
            novo['cs'][i][j] += TF * (teacher['cs'][i][j] - novo['cs'][i][j])
            # Atualiza larguras
            novo['ss'][i][j] += TF * (teacher['ss'][i][j] - novo['ss'][i][j])

            # Limites
            novo['cs'][i][j] = np.clip(novo['cs'][i][j], xmin[i], xmax[i])
            novo['ss'][i][j] = np.clip(
                novo['ss'][i][j], 1e-6, xmax[i] - xmin[i]
            )

    return novo

# -------------------------
# Interface principal
# -------------------------
def mutacao(indv, pop, xmax, xmin):
    if MUTATION_TYPE == "classic":
        return classic_mutation(indv, xmax, xmin)

    # melhor indivíduo = menor erro
    teacher = min(pop, key=lambda x: x['fitness'])

    if np.random.rand() < ALPHA:
        return tlbm_mutation(indv, teacher, xmax, xmin)
    else:
        return classic_mutation(indv, xmax, xmin)
