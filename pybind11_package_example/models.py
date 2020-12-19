import numpy as np
from . import _cpp  

def matrix_vector_product(M, v):
    ''' Perform a matrix-vector product '''
    M = np.require(M, np.float32, 'C')          # Ensure M is a continuous float array
    v = np.require(v, np.float32, 'C')          # Ensure v is a continuous float array
    assert M.shape[1] == v.shape[0]             # Perform checks and conversion in Python

    x = np.zeros(M.shape[0], dtype=np.float32)  # Allocating output array in Python
    _cpp.matrix_vector_product(M, v, x)         # Call the function in C++
    return x                                    # Return the output array
