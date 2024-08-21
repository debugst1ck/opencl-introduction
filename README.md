# OpenCL Introduction
## What is OpenCL?
OpenCL (Open Computing Language) is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators. OpenCL includes a language (based on C99) for writing kernels (functions that execute on OpenCL devices), plus APIs that are used to define and then control the platforms. OpenCL provides parallel computing using task-based and data-based parallelism.

## Compile OpenCL program
To compile in a build directory:

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Make sure you have OpenCL installed on your system. Windows users will have a `.dll` file in the system directory, while Linux users will have a `.so` file in the system32 library directory. If you don't have OpenCL installed, you can install it by following the instructions below.

```bash
$ sudo apt-get install ocl-icd-opencl-dev
```

On Windows, you can install OpenCL by downloading the latest driver for your GPU from the manufacturer's website.

## Run OpenCL program
To run the compiled program:

```bash
$ ./CAB401
```