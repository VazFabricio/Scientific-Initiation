import numpy as np

# x_n é o "estado" ou "memória" do mapa caótico.
x_n = 0.1 
r = 3.7 

def get_valor_caotico():
    """
    Gera o próximo valor caótico usando o Logistic Map.
    Retorna um valor sempre entre 0.0 e 1.0.
    """
    global x_n
    
    # Equação do Logistic Map: x(n+1) = r * x(n) * (1 - x(n))
    x_n = r * x_n * (1.0 - x_n)
    
    if x_n == 0.0 or x_n == 0.5 or x_n == 1.0:
        x_n = np.random.rand()
        
    return x_n

