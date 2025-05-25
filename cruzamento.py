def cruzamento(pai1, pai2, p1, p2):
    nin, nfp1 = len(pai1['cs']), len(pai1['cs'][0])
    nfp2 = len(pai2['ss'][0])
    j2 = p1 + 1

    filho = {
        'cs': [row[:] for row in pai1['cs']],
        'ss': [row[:] for row in pai1['ss']],
        'nfps': None
    }

    for j in range(p1):
        for i in range(nin):
            filho['cs'][i][j] = pai1['cs'][i][j]
            filho['ss'][i][j] = pai1['ss'][i][j]

    for j in range(p2 + 1, nfp2):
        for i in range(nin):
            if j2 >= len(filho['cs'][0]):
                # Expand columns if necessary
                for row in filho['cs']:
                    row.append(0)
                for row in filho['ss']:
                    row.append(0)
            filho['cs'][i][j2] = pai2['cs'][i][j]
            filho['ss'][i][j2] = pai2['ss'][i][j]
        j2 += 1

    filho['nfps'] = len(filho['cs'][0])

    return filho