#pragma once

#include "cuda_runtime.h"

#include "Size.h"
#include "Index.h"

namespace cudacpp {



template<int DIM_GRID, int DIM_BLOCK, int DIM_DATA>
struct GridInfo {
	//using Index = Index<DIM_DATA>;

	Size<DIM_DATA> dataSize;

	__device__ __inline__ auto index() const {
		return CreateIndex<DIM_DATA, DIM_GRID, DIM_BLOCK>();
	}

	__device__ __inline__ bool inRange(const Index<DIM_DATA>& i) const {
		return i.inRange(dataSize);
	}
};



template<int DIM_BLOCKS, int DIM_THREADS, int DIM_DATA>
struct Grid {
	Size<DIM_BLOCKS> blocks;
	Size<DIM_THREADS> threads;
	Size<DIM_DATA> dataSize;

	auto info() const {
		return GridInfo<DIM_BLOCKS, DIM_THREADS, DIM_DATA>{dataSize};
	}


	/*
	template<int DIM_SZ_BL, int DIM_SZ_TH>
	Grid(const Size<DIM_SZ_BL> &b, const Size<DIM_SZ_TH> &t)
		: blocks(b)
		, threads(t)
	{}

	template<int DIM_SZ_BL, int DIM_SZ_TH>
	Grid(const Grid<DIM_SZ_BL, DIM_SZ_TH> &g)
		: blocks(g.blocks)
		, threads(g.threads)
	{}
	*/

	constexpr static int DimBlocks = DIM_BLOCKS;
	constexpr static int DimThreads = DIM_THREADS;

	__host__ __device__ __inline__ DimSize totalBlocks() const { return blocks.volume(); }
	__host__ __device__ __inline__ DimSize threadsPerBlock() const { return threads.volume(); }
	__host__ __device__ __inline__ DimSize totalThreads() const { return totalBlocks() * threadsPerBlock(); }
};


template<int DIM_GRID, int DIM_BLOCK, int DIM_DATA>
static inline auto CreateGrid(const Size<DIM_BLOCK> &szBlock, const Size<DIM_DATA> &szData) {
	return Grid<DIM_GRID, DIM_BLOCK, DIM_DATA>{ szData.divideMin(szBlock), szBlock, szData };
}



}
