# Arquivo: gerarnovapop.py

import numpy as np
import copy # Importante: necessário para copiar os indivíduos

# Supondo que seus outros arquivos se chamam torneio.py, cruzamento.py e mutacao.py
import torneio 
import cruzamento
import mutacao

def gerarnovapop(pop, melhorindvidx, tamPop, taxaCruza, taxaMuta, xmax, xmin):
   
    melhorindv = pop[melhorindvidx]
    nova_pop = [melhorindv]

    while len(nova_pop) < tamPop:

        parent1_idx = torneio.torneio(pop, 3)
        parent2_idx = torneio.torneio(pop, 3)
        parent1 = pop[parent1_idx]
        parent2 = pop[parent2_idx]

        limitep1 = round((parent1['nfps'] - 1) / 2)
        limitep2 = round((parent2['nfps'] - 1) / 2) + 1
        
        limitep1 = max(0, limitep1)
        limitep2 = max(0, limitep2)

        pontocruza1 = round(1 + np.random.rand() * limitep1)
        pontocruza2 = round(limitep2 + np.random.rand() * (parent2['nfps'] - 1))
        
        # --- Crossover ---
        if np.random.rand() < taxaCruza:
            # Cria o primeiro filho
            filho1 = cruzamento.cruzamento(parent1, parent2, int(pontocruza1), int(pontocruza2))

            if np.random.rand() < taxaMuta:
                filho1 = mutacao.mutacao(filho1, xmax, xmin)
            
            nova_pop.append(filho1)

            if len(nova_pop) < tamPop:
                
                filho2 = copy.deepcopy(filho1) 
                
                if np.random.rand() < taxaMuta:
                
                    filho2 = mutacao.mutacao(filho2, xmax, xmin)
                
                nova_pop.append(filho2) 

        # --- Sem Crossover (Seleção) ---
        else:
            # Pega o pai 1 (como referência)
            ind1 = parent1 
            if np.random.rand() < taxaMuta:
    
                ind1 = mutacao.mutacao(ind1, xmax, xmin)
            
            nova_pop.append(ind1)

            # Verifica se ainda há espaço
            if len(nova_pop) < tamPop:
                # Pega o pai 2 (como referência)
                ind2 = parent2
                if np.random.rand() < taxaMuta:
                    # Se mutar, 'ind2' recebe o NOVO objeto
                    ind2 = mutacao.mutacao(ind2, xmax, xmin)
                
                nova_pop.append(ind2)

    # Garante que a população tenha exatamente o tamanho tamPop
    return nova_pop[:tamPop]