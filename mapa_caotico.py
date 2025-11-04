_default = None

class ChaoticRNG:
    def __init__(self, x0=0.123456789, r=3.99):
        self.x = float(x0)
        self.r = float(r)

    def next(self):
        # iterar mapa logístico e retornar novo estado (no intervalo (0,1))
        self.x = self.r * self.x * (1.0 - self.x)
        # proteger limites numéricos
        if self.x <= 0.0:
            self.x = 1e-12
        if self.x >= 1.0:
            self.x = 1 - 1e-12
        return self.x

    def rand(self, a=0.0, b=1.0):
        return a + (b - a) * self.next()




def init(x0=0.123456789, r=3.99):
    """Inicializa o gerador caótico padrão (chamar antes de usar get_valor_caotico)."""
    global _default
    _default = ChaoticRNG(x0=x0, r=r)

def x_inicial():
    return _default.x

def get_valor_caotico():
    """Retorna próximo valor em (0,1) — inicialize com init(...) antes."""
    global _default
    if _default is None:
        init()   # inicializa com valores default se não chamado
    return _default.next()


def rand(a=0.0, b=1.0):
    """Retorna valor mapeado para [a,b]."""
    global _default
    if _default is None:
        init()
    return _default.rand(a, b)
