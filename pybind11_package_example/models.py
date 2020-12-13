import numpy as np
from . import _cpp

def ensure_f32c(func):
    '''ensure input matrix x is C-compatible'''
    mod = lambda x: np.ascontiguousarray(x, dtype=np.float32) if type(x) is np.array else x
    def wrapper(*args, **kwargs):
        args = [mod(v) for v in args]
        kwargs = {k: mod(v) for k,v in kwargs}
        return func(*args, **kwargs)

    return wrapper

@ensure_f32c
def matrix_vector_product(M, v):
    ''' solve Ax=b by the coordinate descent method '''
    assert M.shape[1] == v.shape[0]
    x = np.zeros(M.shape[0], dtype=np.float32)
    _cpp.matrix_vector_product(M, v, x)
    return x
