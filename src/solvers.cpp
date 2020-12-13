float sdot(const float *x, const float *y, int l)
{
        float s = 0;
        int m = l-4;
        int i;
        for (i = 0; i < m; i += 5)
                s += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] +
                        x[i+3] * y[i+3] + x[i+4] * y[i+4];

        for ( ; i < l; i++)        /* clean-up loop */
                s += x[i] * y[i];

        return s;
}

void matrix_vector_product(
        int m, int n, 
        const float *__restrict M, const float *__restrict v, float *__restrict x)
{
    for (int i=0; i<m; i++)
        x[i] = sdot(M+i*n, v, n);
}
