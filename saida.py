import skfuzzy as fuzz
import numpy

def saida(x_orig, c, s, p, q, m):
    
    x = numpy.asarray(x_orig)
    c = numpy.asarray(c) 
    s = numpy.asarray(s)
    p = numpy.asarray(p)
    q = numpy.asarray(q)


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

    for i in range(np): #Para cada linha da base
        a = 0.0
        b = 0.0
        y_atual = numpy.zeros(m)
        w_atual = numpy.ones(m)
        
        for j in range(m): #Para cada função de pertinencia 
            
            y_atual[j] = q[j]
            w_atual[j] = 1.0 
            
            for k in range(n): #Para cada variável
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

# import skfuzzy as fuzz
# import numpy as np

# def saida(x_orig, c, s, p, q, m):
#     x = np.asarray(x_orig)
#     c = np.asarray(c) 
#     s = np.asarray(s)

#     if x.ndim == 0:
#         x = x.reshape(1, 1)
#     elif x.ndim == 1:
#         x = x.reshape(1, -1)
#     elif x.ndim == 2:
#         pass
#     else:
#         raise ValueError("A entrada x_orig deve ser um escalar, um array 1D ou um array 2D.")

#     npontos, n = x.shape
#     ys = np.zeros(npontos)
    
#     w_final = np.ones(m)
#     y_final = np.zeros(m)
#     b_final = 0.0

#     for i in range(npontos):
#         a = 0.0
#         b = 0.0
#         y_atual = np.zeros(m)
#         w_atual = np.ones(m)
        
#         for j in range(m):
#             y_atual[j] = q[j]
#             w_atual[j] = 1.0 
            
#             for k in range(n):
#                 gauss = fuzz.gaussmf(x[i, k], mean=c[k, j], sigma=s[k, j])
                
#                 # if not np.isfinite(gauss):
#                 #     print(f"[DEBUG] gaussmf retornou valor inválido: "
#                 #           f"x={x[i,k]}, mean={c[k,j]}, sigma={s[k,j]}, gauss={gauss}")
                
#                 w_atual[j] *= gauss
#                 print("yatual[j]: ", y_atual[j])
#                 y_atual[j] += p[k, j] * x[i, k]

#             if not np.isfinite(w_atual[j]):
#                 print(f"[DEBUG] w_atual inválido na regra {j}: {w_atual[j]}")

#             if not np.isfinite(y_atual[j]):
#                 print(f"[DEBUG] y_atual inválido na regra {j}: {y_atual[j]}, "
#                       f"q={q[j]}, p={p[:,j]}, x={x[i,:]}")

#             a += w_atual[j] * y_atual[j]
#             b += w_atual[j]

#         if not np.isfinite(a) or not np.isfinite(b):
#             print(f"[DEBUG] acumuladores inválidos: a={a}, b={b}")

#         if b == 0:
#             print("[DEBUG] b==0, substituindo por 1.0")
#             b = 1.0
            
#         ys[i] = a / b

#         if not np.isfinite(ys[i]):
#             print(f"[DEBUG] ys inválido: ys={ys[i]}, a={a}, b={b}")

#         w_final = w_atual
#         y_final = y_atual
#         b_final = b
            
#     return ys, w_final, y_final, b_final
