
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cudacpp\DeviceVector.h"
#include "cudacpp\Index.h"
#include "cudacpp\Grid.h"

#include "cudacpp\DevicePtr.h"
#include "cudacpp\DeviceMemory.h"
#include "cudacpp\DeviceObject.h"

using namespace cudacpp;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

template<int DIM_BLOCKS, int DIM_THREADS, typename T>
__global__ void addKernel(cudacpp::DeviceVector<T> c, const cudacpp::DeviceVector<T> a, const cudacpp::DeviceVector<T> b)
{
	auto idx = cudacpp::Index<1>::create<DIM_BLOCKS, DIM_THREADS>();
	if (idx.inRange(c.size())) {
		c[idx] = a[idx] + b[idx];
	}
}

__device__ __inline__ int my1DimIndex() {
	int blockId = blockIdx.x 
		        + blockIdx.y * gridDim.x
		        + blockIdx.z * gridDim.x * gridDim.y;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		           + threadIdx.x
		           + threadIdx.y * blockDim.x
		           + threadIdx.z * blockDim.x * blockDim.y;
	return threadId;
}

template<typename T, typename INFO>
__global__ void addKernel2(DevicePtr<T> c, DevicePtr<T const> a, DevicePtr<T const> b, INFO info )
{
	//int myidx = my1DimIndex();
	auto myIdx = info.index();
	if (info.inRange(myIdx)) {
		c[myIdx] = a[myIdx] + b[myIdx];
	}
}

template<typename T>
struct Op {
	virtual __device__ __host__ T operator()(T t1, T t2) const = 0;
};


struct OpPlus : public Op<int> {
	int *_i;

	__device__ __host__ ~OpPlus() { delete _i; }

	__device__ __host__ OpPlus(int i) : _i(new int(i)) {}

	__device__ __host__ int operator()(int t1, int t2) const override {
		return t1 + t2 + *_i;
	}
};

/*
template<typename T, typename INFO>
__global__ void addKernelOp(DevicePtr<T> c, DevicePtr<T const> a, DevicePtr<T const> b, INFO info, DevicePtr<OpPlus> op)
{
	//int myidx = my1DimIndex();
	auto myIdx = info.index();
	if (info.inRange(myIdx)) {
		int aaa = op->operator()(2, 3);
		c[myIdx] = op->operator()(a[myIdx], b[myIdx]);
		//c[myIdx] = (*op)(a[myIdx], b[myIdx]);
		//c[myIdx] = a[myIdx] + b[myIdx];
	}
}
*/

template<int DIM_BLOCKS, int DIM_THREADS, typename T>
__global__ void addKernelOp(cudacpp::DeviceVector<T> c, const cudacpp::DeviceVector<T> a, const cudacpp::DeviceVector<T> b, DevicePtr<Op<T>> op)
{
	auto idx = cudacpp::Index<1>::create<DIM_BLOCKS, DIM_THREADS>();
	if (idx.inRange(c.size())) {
		c[idx] = (*op)(a[idx], b[idx]);
	}
}

template<int DIM_BLOCKS, int DIM_THREADS, typename T>
__global__ void addKernelLambda(cudacpp::DeviceVector<T> c, const cudacpp::DeviceVector<T> a, const cudacpp::DeviceVector<T> b)
{
	auto idx = cudacpp::Index<1>::create<DIM_BLOCKS, DIM_THREADS>();
	if (idx.inRange(c.size())) {
		auto op = [&] { return a[idx] + b[idx]; };
		c[idx] = op();
	}
}


int basics(int size, const int *a, const int *b, int *c)
{
	auto pa = MakeDevicePtr(a);
	auto SIZE = unsigned(32);

	int *cDev;
	cudaMalloc(&cDev, SIZE);
	cudaMemcpy(cDev, c, SIZE, cudaMemcpyHostToDevice);

	//DevicePtr<int> bDev;
	//cudaMalloc(&bDev, SIZE);

	Size<1> dataSize{ (unsigned int)size };
	Size<1> blockSize{ (unsigned int)size };
	auto grid = CreateGrid<1>(blockSize, dataSize);

//	DeviceObject<OpPlus> op(42);

//	addKernelOp<<<1, size>>>(MakeDevicePtr(cDev), pa, MakeDevicePtr(b), grid.info(), (DevicePtr<OpPlus>)op);



	//addKernel2<<<grid.blocks, grid.threads>>>(MakeDevicePtr(cDev), pa, MakeDevicePtr(b), grid.info());

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
	auto dev_a = cudacpp::DeviceMemory<int>::AllocateElements(size);
	auto dev_b = cudacpp::DeviceMemory<int>::AllocateElements(size);
	auto dev_c = cudacpp::DeviceMemory<int>::AllocateElements(size);

    // Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudacpp::CopyElements(dev_a, a, size);//   cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    cudaStatus = cudacpp::CopyElements(dev_b, b, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.

	cudacpp::DeviceVector<int> vec_a{ dev_a, size };

	auto grid = cudacpp::CreateGrid<1>(cudacpp::Size<1>{4}, cudacpp::Size<1>{vec_a.size()});
	     grid = cudacpp::CreateGrid<1, 1>(4, vec_a.size());


	 DeviceObject<OpPlus> op(42);


    //addKernel<grid.DimBlocks, grid.DimThreads, int><<<grid.blocks, grid.threads>>>(dev_c, vec_a, dev_b);
	 //addKernelOp<grid.DimBlocks, grid.DimThreads, int> << <grid.blocks, grid.threads >> >(dev_c, vec_a, dev_b, (DevicePtr<OpPlus>)op);
	 addKernelLambda<grid.DimBlocks, grid.DimThreads, int><<<grid.blocks, grid.threads>>>(dev_c, vec_a, dev_b);

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
	cudaStatus = cudacpp::CopyElements(c, dev_c, size); // cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
    }

    return cudaStatus;
}
