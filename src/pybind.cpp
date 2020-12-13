#if USE_PYTORCH
    #include <torch/extension.h>
    using Tensor=torch::Tensor;
    float *FPTR(Tensor& a) { return a.data_ptr<float>(); }
    int   *IPTR(Tensor& a) { return a.data_ptr<int>();   }
#else
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
    using Tensor = py::array;
    float *FPTR(Tensor& a) { return (float*) a.mutable_data(); }
    int   *IPTR(Tensor& a) { return (int*)   a.mutable_data(); }
#endif

#include "solvers.h"

void py_matrix_vector_product(Tensor M, Tensor v, Tensor x)
{
    int m = M.shape(0);
    int n = M.shape(1);

    matrix_vector_product(m, n, FPTR(M), FPTR(v), FPTR(x));
}


PYBIND11_MODULE(EXTENSION_NAME, m) {
    m.def("matrix_vector_product", &py_matrix_vector_product);
}
