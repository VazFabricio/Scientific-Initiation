import numpy as np
import torneio
import cruzamento
import mutacao
import copy

def gerarnovapop(pop, melhorindvidx, tamPop, taxaCruza, taxaMuta, xmax, xmin):
    
    melhorindv = pop[melhorindvidx]

    nova_pop = [copy.deepcopy(melhorindv)]

    while len(nova_pop) < tamPop:
        z = len(nova_pop)

        parent1 = torneio.torneio(pop, 3)
        parent2 = torneio.torneio(pop, 3)

        limitep1 = round((pop[parent1]['nfps'] - 1) / 2)
        limitep2 = round((pop[parent2]['nfps'] - 1) / 2) + 1

        pontocruza1 = round(1 + np.random.rand() * limitep1)
        pontocruza2 = round(limitep2 + np.random.rand() * (pop[parent2]['nfps'] - 1))

        if np.random.rand() < taxaCruza:
            filho = cruzamento.cruzamento(pop[parent1], pop[parent2], pontocruza1, pontocruza2)
            
            if np.random.rand() < taxaMuta:
                filho = mutacao.mutacao(filho, xmax, xmin)
            
            nova_pop.append(filho)

            if len(nova_pop) < tamPop and np.random.rand() < taxaMuta:
                filho2 = mutacao.mutacao(copy.deepcopy(filho), xmax, xmin)
                nova_pop.append(filho2)
        else:

            ind1 = copy.deepcopy(pop[parent1])
            if np.random.rand() < taxaMuta:
                ind1 = mutacao.mutacao(ind1, xmax, xmin)
            nova_pop.append(ind1)

            if len(nova_pop) < tamPop:
                ind2 = copy.deepcopy(pop[parent2])
                if np.random.rand() < taxaMuta:
                    ind2 = mutacao.mutacao(ind2, xmax, xmin)
                nova_pop.append(ind2)

    return nova_pop