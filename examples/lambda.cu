
#include "cuda_runtime.h"
#include <stdio.h>

#include <nvfunctional>
#include <functional>

#include "cudacpp\DeviceVector.h"


__device__ __inline__ void SetVal(int &i, int val) {
	i = val;
}

template<typename T>
__global__ void applyKernel(cudacpp::DeviceVector<int> c, T op)
{
	auto idx = threadIdx.x;
	op(c[idx]);
}

template<typename T>
__global__ void applyKernelGetFunc(cudacpp::DeviceVector<int> c, T op)
{
	auto idx = threadIdx.x;
	op.getFunc()(c[idx]);
}


__global__ void setValueInnerLambda(cudacpp::DeviceVector<int> c, int val)
{
	auto idx = threadIdx.x;
	auto op = [=](int& i) { i = val; };
	op(c[idx]);
}



struct AddValue {
	int _val;

	AddValue(int val) : _val(val) {}

	nvstd::function<void(int&)> __device__ getFunc() {
		return [*this] __device__ (int& i) {
			i = _val;
		};
	}

	void doApplyKernel(cudacpp::DeviceMemoryT<int>& dev_c) {
		auto f = [*this] __device__ (int& i) {
			i = _val;
		};
		applyKernel<<<1, dev_c.sizeElements()>>>(dev_c, f);
	}
};



int testLambda(int size, int *c, int val) {
	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output) 
	cudacpp::DeviceMemoryT<int> dev_c(size);

	// VERSION 1
	//setValueInnerLambda<<<1, size>>>(dev_c, val);

	// VERSION 2
	//auto op = [=] __device__ __host__(int& v) { v = val; };
	//applyKernel<<<1, size>>>(dev_c, op);

	AddValue addF{ val };
	// VESRION 3
	//addF.doApplyKernel(dev_c);
	
	// VERSION 4
	applyKernelGetFunc<<<1, size>>>(dev_c, addF);


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


	return 0;
}