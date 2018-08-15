
#include "cuda_runtime.h"
#include <stdio.h>

#include <nvfunctional>
#include <functional>

#include "cudacpp\DeviceVector.h"
#include "cudacpp\DeviceMemory.h"


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
__global__ void applyKernel2(cudacpp::DeviceVector<int> c, T op)
{
	auto idx = threadIdx.x;
	op(c[idx], 4);
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
	auto op = [=](auto& i) { i = val; };
	op(c[idx]);

	//c[idx] = [](auto v){retu}
}



struct AddValue {
	int _val;

	AddValue(int val) : _val(val) {}

	nvstd::function<void(int&)> __device__ getFunc() {
		return [*this] (int& i) {
			i = _val;
		};
	}

	void doApplyKernel(cudacpp::DeviceMemory<int> &dev_c) {
		auto f = [*this] __device__ (int& i) {
			i = _val;
		};
		applyKernel<<<1, dev_c.sizeElements()>>>(dev_c, f);
	}
};


struct A {
	//A();
	int a;
	auto f() {
//		return [this]__device__(int& val) {val = a + 1; };
	}
	//virtual ~A();
};


template<typename OP>
__global__ void applyKernelOp(cudacpp::DeviceVector<int> c, cudacpp::DeviceVector<int> a, cudacpp::DeviceVector<int> b, OP op)
{
	auto idx = threadIdx.x;
	c[idx] = op.op()(a[idx], b[idx]);
}


__global__ void applyKernelDirect(cudacpp::DeviceVector<int> c, cudacpp::DeviceVector<int> a, cudacpp::DeviceVector<int> b, int val)
{
	auto idx = threadIdx.x;
	c[idx] = a[idx] + b[idx] + val;
}



struct FNC {
	int _i;
	FNC(int i) : _i(i) {}

	nvstd::function<int(int, int)> __device__ op() {
		return [*this](auto a, auto b){ return a + b + _i; };
	}
};


int testLambda1() {
	auto dev_c = cudacpp::DeviceMemory<int>::AllocateElements(1);
	auto dev_a = cudacpp::DeviceMemory<int>::AllocateElements(1);
	auto dev_b = cudacpp::DeviceMemory<int>::AllocateElements(1);
	int num = 10;
	auto cudaStatus = cudacpp::CopyElements(dev_a, &num, 1);
	cudaStatus = cudacpp::CopyElements(dev_b, &num, 1);

	//applyKernelOp<<<1, 1>>>(dev_c, []__device__{return 42;});
	FNC fnc{ 42 };
	//fnc.apply(dev_c);
	applyKernelOp<<<1, 1>>>(dev_c, dev_a, dev_b, fnc);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	int i;
	cudaStatus = cudacpp::CopyElements(&i, dev_c, 1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	printf("i=%d\n", i);

	applyKernelDirect << <1, 1 >> >(dev_c, dev_a, dev_b, 42);

}


int testLambda(int size, int *c, int val) {
	testLambda1();


	cudaError_t cudaStatus;


	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output) 
	auto dev_c = cudacpp::DeviceMemory<int>::AllocateBytes(size);

	// VERSION 1
	setValueInnerLambda<<<1, size>>>(dev_c, val);


	A aaa;

	// VERSION 2
	auto op = [=] __device__(auto& v) { v = size; };
	//auto op = aaa.f();
	applyKernel<<<1, size>>>(dev_c, op);

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
	cudaStatus = cudacpp::CopyElements(c, dev_c, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	return cudaStatus;
}