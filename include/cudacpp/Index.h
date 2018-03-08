#pragma once

#include "cuda_runtime.h"


namespace cudacpp {

using DimSize = unsigned int;

template<int DIM>
struct Size;

template<>
struct Size<0> {
	__host__ __device__ __inline__ DimSize volume() const { return 1; }
	__host__ __device__ __inline__ operator dim3() const { return{ 1, 1, 1 }; }
};

template<>
struct Size<1> {
	DimSize x;
	Size(DimSize xx) : x(xx) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x; }
	__host__ __device__ __inline__ DimSize volume() const { return x; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, 1, 1 }; }
};

template<>
struct Size<2> : public Size<1> {
	DimSize y;
	Size(DimSize xx, DimSize yy) : Size<1>(xx), y(yy) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x && y < sz.y; }
	__host__ __device__ __inline__ DimSize volume() const { return Size<1>::volume() * y; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, y, 1 }; }
};

template<>
struct Size<3> : public Size<2> {
	DimSize z;
	Size(DimSize xx, DimSize yy, DimSize zz) : Size<2>(xx, yy), z(zz) {}
	Size(dim3 d3) : Size(d3.x, d3.y, d3.z) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x && y < sz.y && z < sz.z; }
	__host__ __device__ __inline__ DimSize volume() const { return Size<2>::volume() * z; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, y, z }; }
};


template<int DIM_BLOCKS, int DIM_THREADS>
struct GridSize {
	Size<DIM_BLOCKS> blocks;
	Size<DIM_THREADS> threads;

	__host__ __device__ __inline__ DimSize totalBlocks() const { return blocks.volume(); }
	__host__ __device__ __inline__ DimSize threadsPerBlock() const { return threads.volume(); }
	__host__ __device__ __inline__ DimSize totalThreads() const { return totalBlocks() * threadsPerBlock(); }
};

template<int DIM>
struct Index;

template<>
struct Index<1> {
	DimSize x;

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
};

template<>
struct Index<2> {
	DimSize x;
	DimSize y;

	template<int DIM_BLOCKS, int DIM_THREADS>
	__host__ __device__ __inline__ static Index create();

	template<>
	__host__ __device__ __inline__ static Index create<0, 2>() {
		return {threadIdx.x, threadIdx.y};
	}

	template<>
	__host__ __device__ __inline__ static Index create<2, 2>() {
		return { (blockIdx.x * blockDim.x) + threadIdx.x, (blockIdx.y * blockDim.y) + threadIdx.y };
	}
};

}
