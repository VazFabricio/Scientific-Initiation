import skfuzzy as fuzz
import numpy
#pop(z).saida=saida(xt,pop(z).cs,pop(z).ss,p,q,pop(z).nfps)
def saida(x, c, s, p, q, m):
    np, n = (x).shape
    ys = numpy.zeros(np)
    for i in range(np):
        a = 0
        b = 0
        y = numpy.zeros(m)
        w = numpy.ones(m)
        for j in range(m):
            y[j] = q[j]
            w[j] = 1
            for k in range(n):
                w[j] *= fuzz.gaussmf(x[i,k], mean = c[k,j], sigma = s[k,j])

                y[j] += p[k,j] * x[i,k]

            a += w[j]*y[j]
            b += w[j]

        if(b == 0):
            b = 1
        ys[i] = a/b

        return ys
