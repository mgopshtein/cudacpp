#pragma once

#include "cuda_runtime.h"
#include <utility>

#include "Size.h"
#include "Index.h"

namespace cudacpp {


class DeviceMemory {
	void *_p = nullptr;
	std::size_t _len = 0;

public:
	explicit DeviceMemory(std::size_t len) 
		: _len(len)
	{
		cudaMalloc(&_p, len);
	}

	DeviceMemory(DeviceMemory &&from)
		: _p(from._p)
		, _len(from._len)
	{
		from._p = nullptr;
	}

	~DeviceMemory() {
		if (_p) {
			cudaFree(_p);
		}
	}

	DeviceMemory(const DeviceMemory&) = delete;
	DeviceMemory& operator=(const DeviceMemory&) = delete;
	DeviceMemory& operator=(DeviceMemory&&) = delete;

	operator void *() { return _p; }
	operator const void *() const { return _p; }
	void * ptr() { return _p; }
	const void * ptr() const { return _p; }
	auto sizeBytes() const { return _len; }
};


template<typename T>
class DeviceMemoryT
	: public DeviceMemory
{
	DimSize _sz;

public:
	DeviceMemoryT(DimSize sz)
		: DeviceMemory(sizeof(T) * sz)
		, _sz(sz)
	{}

	DeviceMemoryT(DeviceMemoryT<T> &&from)
		: DeviceMemory(std::move(from))
		, _sz(from._sz)
	{}

	operator T *() { return reinterpret_cast<T*>(ptr()); }
	operator const T *() const { return reinterpret_cast<T*>(ptr()); }

	auto sizeElements() const { return _sz; }
};


template<typename T>
class DeviceVector {
	T *_p;
	Size<1> _sz;

public:
	DeviceVector(DeviceMemory &mem, DimSize sz)
		: _p(reinterpret_cast<T*>(mem.ptr()))
		, _sz(sz)
	{}

	DeviceVector(DeviceMemoryT<T> &mem)
		: _p(mem)
		, _sz(mem.sizeElements())
	{}

	DeviceVector() = delete;

	__device__ __host__ __inline__ T& operator[](std::size_t i) { return _p[i]; }
	__device__ __host__ __inline__ T& operator[](Index<1> i) { return _p[i.x]; }
	__device__ __host__ __inline__ const T& operator[](std::size_t i) const { return _p[i]; }
	__device__ __host__ __inline__ const T& operator[](Index<1> i) const { return _p[i.x]; }
	__device__ __host__ __inline__ operator T* () { return _p; }
	__device__ __host__ __inline__ auto size() { return _sz; }
};


}

