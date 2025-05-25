import skfuzzy as fuzz
import numpy

def saida(x_orig, c, s, p, q, m):
    x = numpy.asarray(x_orig)
    c = numpy.asarray(c) 
    s = numpy.asarray(s)

    if x.ndim == 0:
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 2:
        pass
    else:
        raise ValueError("A entrada x_orig deve ser um escalar, um array 1D ou um array 2D.")

    np, n = x.shape

    ys = numpy.zeros(np)
    
    w_final = numpy.ones(m)
    y_final = numpy.zeros(m)
    b_final = 0.0

    for i in range(np):
        a = 0.0
        b = 0.0
        y_atual = numpy.zeros(m)
        w_atual = numpy.ones(m)
        
        for j in range(m):
            y_atual[j] = q[j]
            w_atual[j] = 1.0 
            
            for k in range(n):
                w_atual[j] *= fuzz.gaussmf(x[i,k], mean = c[k,j], sigma = s[k,j])
                y_atual[j] += p[k,j] * x[i,k]
            
            a += w_atual[j] * y_atual[j]
            b += w_atual[j]

        if b == 0:
            b = 1.0
            
        ys[i] = a / b

        w_final = w_atual
        y_final = y_atual
        b_final = b
            
    return ys, w_final, y_final, b_final
