#pragma once

#include "cuda_runtime.h"

#include "Size.h"

namespace cudacpp {


template<int DIM>
struct Index {
	template<int DIM_BLOCKS, int DIM_THREADS>
	__host__ __device__ __inline__ static Index create();
};

template<>
struct Index<1> 
	: public Size<1>
{
	using Size<1>::Size;

	template<int DIM_BLOCKS, int DIM_THREADS>
	__host__ __device__ __inline__ static Index create();

	template<>
	__host__ __device__ __inline__ static Index create<0, 1>()
	{ return { threadIdx.x }; }

	template<>
	__host__ __device__ __inline__ static Index create<1, 1>()
	{ return { (blockIdx.x * blockDim.x) + threadIdx.x }; }

	template<>
	__host__ __device__ __inline__ static Index create<2, 1>()
	{
		int blockId = blockIdx.y * gridDim.x + blockIdx.x;
		return {blockId * blockDim.x + threadIdx.x};
	}

	template<>
	__host__ __device__ __inline__ static Index create<3, 1>() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
		return {blockId * blockDim.x + threadIdx.x};
	}

	__host__ __device__ __inline__ bool inRange(Size<1> sz) const { return *this < sz; }

	template<int DIM_BLOCKS, int DIM_THREADS>
	__device__ __inline__ static Index create(Size<DIM_BLOCKS>, Size<DIM_THREADS>) {
		return create<DIM_BLOCKS, DIM_THREADS>();
	}

};

template<>
struct Index<2>
	: public Size<2>
{
	using Size<2>::Size;

	template<int DIM_BLOCKS, int DIM_THREADS>
	__device__ __inline__ static Index create();

	template<>
	__device__ __inline__ static Index create<0, 2>() {
		return {threadIdx.x, threadIdx.y};
	}

	template<>
	__device__ __inline__ static Index create<2, 2>() {
		return { (blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y };
	}

	__device__ __inline__ bool inRange(Size<2> sz) const { return *this < sz; }

	template<int DIM_BLOCKS, int DIM_THREADS>
	__device__ __inline__ static Index create(Size<DIM_BLOCKS>, Size<DIM_THREADS>) {
		return create<DIM_BLOCKS, DIM_THREADS>();
	}
};


template<int DIM, int DIM_BLOCKS, int DIM_THREADS>
__device__ __inline__ static auto CreateIndex() {
	return Index<DIM>::create(Size<DIM_BLOCKS>{0}, Size<DIM_THREADS>{0});
}


}



