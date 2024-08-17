__kernel void matrix_mul(
    __global float *A, __global float *B,
    __global float *C, const int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < N; k++)
    {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}