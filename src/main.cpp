#include <iostream>
#include <vector>

// Opencl matrix multiplication
#include <CL/cl.hpp>

#include <chrono>
#include <string>
#include <random>

// Sequential matrix multiplication function
void matrix_mul_seq(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[])
{
    // Start the timer
    // Create the input and output buffers
    const int N = 100;
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    // Initialize the input buffers with random values
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // Get the most performant device
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Device device = devices.front();

    // Create a context
    cl::Context context(device);

    // Create a command queue
    cl::CommandQueue queue(context, device);

    // Create a program
    cl::Program::Sources sources;

    std::string kernel_code = R"(
            __kernel void matrix_mul(__global float* A, __global float* B, __global float* C, const int N)
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
        )";

    // This is GEMM1 kernel

    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    cl::Program program(context, sources);

    // Build the program
    program.build("-cl-std=CL1.2");

    // Create a kernel
    cl::Kernel kernel(program, "matrix_mul");

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, N * N * sizeof(float));
    cl::Buffer buffer_B(context, CL_MEM_READ_ONLY, N * N * sizeof(float));
    cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float));

    // Write the input buffers
    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, N * N * sizeof(float), A.data());
    queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, N * N * sizeof(float), B.data());

    // Set the kernel arguments
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    kernel.setArg(2, buffer_C);
    kernel.setArg(3, N);

    // Execute the kernel
    cl::NDRange global(N, N);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);

    // Read the output buffer
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, N * N * sizeof(float), C.data());

    // Clean up
    queue.finish();

    // End the timer
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    // Print the device name
    std::string device_name;
    device.getInfo(CL_DEVICE_NAME, &device_name);
    std::cout << "Device: " << device_name << std::endl;

    // Print the duration
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // Sequential matrix multiplication
    // Clear the output buffer
    std::vector<float> C_seq(N * N, 0.0f);

    // Start the timer
    start = std::chrono::steady_clock::now();

    // Perform the matrix multiplication
    matrix_mul_seq(A, B, C_seq, N);

    // End the timer
    end = std::chrono::steady_clock::now();

    // Calculate the duration
    duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);


    // Print the duration
    std::cout << "Duration: " << duration.count() << " seconds" << std::endl;

    // Print size of the matrices
    std::cout << "Size of matrix A: " << A.size() << std::endl;
    std::cout << "Size of matrix B: " << B.size() << std::endl;
    std::cout << "Size of matrix C: " << C.size() << std::endl;

    // Compare the results
    for (int i = 0; i < N * N; i++)
    {
        if (std::abs(C[i] - C_seq[i]) > 0.0001)
        {
            std::cout << std::abs(C[i] - C_seq[i]) << std::endl;
            std::cout << "Results do not match!" << std::endl;
            break;
        }
    }

    return 0;
}