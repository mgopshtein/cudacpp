#pragma once


#include "cuda_runtime.h"
#include <utility>

#include "DevicePtr.h"
#include "DeviceMemory.h"


namespace cudacpp {



namespace detail_DeviceObject {

template<typename T, typename... ARGS>
__global__ void AllocateObject(DevicePtr<T> p, ARGS... args) {
	new (p) T(args...);
}

template<typename T>
__global__ void DeleteObject(DevicePtr<T> p) {
	p->~T();
}

}

template<typename T>
class DeviceObject {
	DeviceMemory<T> _p;

	template<typename... ARGS>
	static void AllocateObjectOnDevice(DevicePtr<T> p, ARGS... args) {
		detail_DeviceObject::AllocateObject<<<1, 1>>>(p, args...);
		auto res = cudaDeviceSynchronize();
	}

	static void DeleteObjectOnDevice(DevicePtr<T> p) {
		detail_DeviceObject::DeleteObject<<<1, 1>>>(p);
		auto res = cudaDeviceSynchronize();
	}

public:
	template<typename... ARGS>
	DeviceObject(ARGS... args) 
		: _p(DeviceMemory<T>::AllocateElements(1))
	{
		detail_DeviceObject::AllocateObject<T><<<1, 1>>>(_p, args...);
		cudaDeviceSynchronize();
		//AllocateObjectOnDevice(_p, args...);
		//auto res = CopyElements(&_p, _pp, 1);
	}

	DeviceObject(DeviceObject &&from)
		: _p(std::move(from._p))
	{}

	DeviceObject& operator=(DeviceObject &&from) {
		_p = std::move(from._p);
		return *this;
	}

	~DeviceObject() {
		if (_p) {
			//DeleteObjectOnDevice(_p);
			detail_DeviceObject::DeleteObject<T><<<1, 1 >>>(_p);
			auto res = cudaDeviceSynchronize();
		}
	}

	DeviceObject(const DeviceObject&) = delete;
	DeviceObject& operator=(const DeviceObject&) = delete;

public:
	operator DevicePtr<T>() const { 
		return _p;
	}
};


}