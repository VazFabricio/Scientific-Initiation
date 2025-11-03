import numpy

def gaussmf_manual(x, mean, sigma):

    sigma = numpy.where(sigma == 0, 1e-9, sigma)
    
    return numpy.exp(-numpy.power(x - mean, 2.) / (2 * numpy.power(sigma, 2.)))

def saida(x_orig, c, s, p, q, m):
    
    x = numpy.asarray(x_orig)
    c = numpy.asarray(c) 
    s = numpy.asarray(s)
    p = numpy.asarray(p)
    q = numpy.asarray(q)

    if x.ndim == 1:
        x = x.reshape(1, -1)
    

    x_exp = x[:, numpy.newaxis, :]  
    c_exp = c.T[numpy.newaxis, :, :]
    s_exp = s.T[numpy.newaxis, :, :] 

    w_matriz = gaussmf_manual(x_exp, c_exp, s_exp)
    w_final = w_matriz.prod(axis=2)

    y_final = (x @ p) + q

    a = (w_final * y_final).sum(axis=1)
    b = w_final.sum(axis=1)    

    b = numpy.where(b == 0, 1.0, b)
    
    ys = a / b

    return ys, w_final, y_final, b