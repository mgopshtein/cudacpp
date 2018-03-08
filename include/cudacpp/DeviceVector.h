#pragma once

#include "cuda_runtime.h"


namespace cudacpp {


template<typename T>
class DeviceVector {
	T *_p;
	explicit DeviceVector(T *p) : _p(p) {}
	DeviceVector() : _p(nullptr) {}

	// we can add a destructor which will cudaFree the memory
public:
	__device__ __host__ __inline__ T& operator[](std::size_t i) { return _p[i]; }
	__device__ __host__ __inline__ const T& operator[](std::size_t i) const { return _p[i]; }
	__device__ __host__ __inline__ operator T* () { return _p; }

	static DeviceVector allocate(std::size_t sz) {
		DeviceVector res;
		cudaMalloc((void**)&res._p, sizeof(T) * sz);
		return res;
	}

	void release() {
		if (_p) {
			cudaFree(_p);
		}
	}

};


}

