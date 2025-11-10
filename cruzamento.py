def cruzamento(pai1, pai2, p1, p2):
    nin = len(pai1['cs'])
    nfp1 = len(pai1['cs'][0])
    nfp2 = len(pai2['cs'][0])
    j2 = p1

    filho = {
        'cs': [row[:] for row in pai1['cs']],
        'ss': [row[:] for row in pai1['ss']],
        'p': [row[:] for row in pai1['p']],
        'q': list(pai1['q']),
        'nfps': None
    }


    for j in range(p2 + 1, nfp2):
        for i in range(nin):
            if j2 >= len(filho['cs'][0]):
                for row_idx in range(len(filho['cs'])):
                    filho['cs'][row_idx].append(0)
                    filho['ss'][row_idx].append(0)
                    filho['p'][row_idx].append(0)
                
                filho['q'].append(0)

            # Atribuir os valores do pai2 ao filho
            filho['cs'][i][j2] = pai2['cs'][i][j]
            filho['ss'][i][j2] = pai2['ss'][i][j]
            filho['p'][i][j2] = pai2['p'][i][j]
        
        # Atribuir o valor de q fora do loop interno de 'i'
        filho['q'][j2] = pai2['q'][j]
        j2 += 1

    filho['nfps'] = len(filho['cs'][0])

    return filho