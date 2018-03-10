#pragma once

#include "cuda_runtime.h"

#include "Size.h"

namespace cudacpp {


template<int DIM_BLOCKS, int DIM_THREADS>
struct Grid {
	Size<DIM_BLOCKS> blocks;
	Size<DIM_THREADS> threads;

	constexpr static int DimBlocks = DIM_BLOCKS;
	constexpr static int DimThreads = DIM_THREADS;

	__host__ __device__ __inline__ DimSize totalBlocks() const { return blocks.volume(); }
	__host__ __device__ __inline__ DimSize threadsPerBlock() const { return threads.volume(); }
	__host__ __device__ __inline__ DimSize totalThreads() const { return totalBlocks() * threadsPerBlock(); }
};


template<int DIM_DATA>
static inline auto CreateGrid(const Size<DIM_DATA> &th, const Size<DIM_DATA> &sz) {
	return Grid<DIM_DATA, DIM_DATA>{ sz.divideMin(th), th };
}



}
