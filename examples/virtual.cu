
#include "cuda_runtime.h"
#include <stdio.h>

#include "cudacpp\DeviceVector.h"
#include "cudacpp\DeviceMemory.h"


class IntProvider {
public:
	virtual __device__ int getNumber() const = 0;
};


class SimpleIntProvider
	: public IntProvider
{
	int _i;

public:
	__device__ __inline__ SimpleIntProvider(int i) : _i(i) {}

	__device__ int getNumber() const override { return _i; }
};


class SumIntProvider
	: public IntProvider
{
	int _a;
	int _b;

public:
	__device__ __inline__ SumIntProvider(int a, int b) : _a(a), _b(b) {}

	__device__ int getNumber() const override { return _a + _b; }

};


__device__ __inline__ void putValue(int& to, const IntProvider& ip) {
	to = ip.getNumber();
}



__global__ void addKernel(cudacpp::DeviceVector<int> c, int val)
{
	auto idx = threadIdx.x;
	//SimpleIntProvider sip{ val };
	SumIntProvider sip{ val, 2 };
	putValue(c[idx], sip);
}


int testVirtual(int size, int *c, int val) {
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output) 
	auto dev_c = cudacpp::DeviceMemory<int>::AllocateElements(size);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, val);

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
	cudaStatus = CopyElements(c, dev_c, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	return cudaStatus;


	return 0;
}