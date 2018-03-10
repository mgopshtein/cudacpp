
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cudacpp\DeviceVector.h"
#include "cudacpp\Index.h"
#include "cudacpp\Grid.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

template<int DIM_BLOCKS, int DIM_THREADS, typename T>
__global__ void addKernel(cudacpp::DeviceVector<T> c, const cudacpp::DeviceVector<T> a, const cudacpp::DeviceVector<T> b)
{
	auto idx = cudacpp::Index<1>::create<DIM_BLOCKS, DIM_THREADS>();
	if (idx.inRange(c.size())) {
		c[idx] = a[idx] + b[idx];
	}
}

int basics(int size, const int *a, const int *b, int *c)
{
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for three vectors (two input, one output) 
	cudacpp::DeviceMemory dev_a(sizeof(int) * size);
	cudacpp::DeviceMemoryT<int> dev_b(size);
	cudacpp::DeviceMemoryT<int> dev_c(size);

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.

	cudacpp::DeviceVector<int> vec_a{ dev_a, size };

	auto grid = cudacpp::CreateGrid(cudacpp::Size<1>{4}, vec_a.size());

    addKernel<grid.DimBlocks, grid.DimThreads, int><<<grid.blocks, grid.threads>>>(dev_c, vec_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    return cudaStatus;
}
