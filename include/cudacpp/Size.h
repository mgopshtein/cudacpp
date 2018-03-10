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
	__host__ __device__ __inline__ Size<0> divideMin(Size<0>) const { return {}; }
};

template<>
struct Size<1> {
	DimSize x;
	__host__ __device__ __inline__ Size(DimSize xx) : x(xx) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x; }
	__host__ __device__ __inline__ DimSize volume() const { return x; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, 1, 1 }; }

	__host__ __device__ __inline__ Size<1> divideMin(const Size<1> &sz) const { return{ (x + sz.x - 1) / sz.x }; }

};

template<>
struct Size<2> : public Size<1> {
	DimSize y;
	__host__ __device__ __inline__ Size(DimSize xx, DimSize yy) : Size<1>(xx), y(yy) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x && y < sz.y; }
	__host__ __device__ __inline__ DimSize volume() const { return Size<1>::volume() * y; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, y, 1 }; }

	__host__ __device__ __inline__ Size<2> divideMin(const Size<2> &sz) const { return{ (x + sz.x - 1) / sz.x, (y + sz.y - 1) / sz.y }; }

};

template<>
struct Size<3> : public Size<2> {
	DimSize z;
	__host__ __device__ __inline__ Size(DimSize xx, DimSize yy, DimSize zz) : Size<2>(xx, yy), z(zz) {}
	__host__ __device__ __inline__ Size(dim3 d3) : Size(d3.x, d3.y, d3.z) {}

	__host__ __device__ __inline__ bool operator<(const Size& sz) const { return x < sz.x && y < sz.y && z < sz.z; }
	__host__ __device__ __inline__ DimSize volume() const { return Size<2>::volume() * z; }
	__host__ __device__ __inline__ operator dim3() const { return{ x, y, z }; }

	__host__ __device__ __inline__ Size<3> divideMin(const Size<3> &sz) const { return{ (x + sz.x - 1) / sz.x, (y + sz.y - 1) / sz.y, (z + sz.z - 1) / sz.z }; }
};


}
