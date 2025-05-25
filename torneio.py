import random

def torneio(pop, k):
    tamanho_populacao = len(pop)
    indice_selecionado = -1

    for i in range(k):
        indice_sorteado = random.randint(0, tamanho_populacao - 1)
        
        if i == 0:
            indice_selecionado = indice_sorteado
        else:
            if pop[indice_sorteado]['fitness'] < pop[indice_selecionado]['fitness']:
                indice_selecionado = indice_sorteado
                
    return indice_selecionado