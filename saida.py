import skfuzzy as fuzz

#pop(z).saida=saida(xt,pop(z).cs,pop(z).ss,p,q,pop(z).nfps)
def saida(x, c, s, p, q, m):
    np, n = (x).shape
    for i in range(np):
        a = 0
        b = 0
        for j in range(m):
            y[j] = q[j]
            w[j] = 1
            for k in range(n):
                w[j] *= fuzz.gaussmf(x[i,k], mean = c[k,j], sigma = s[k,j])
                