#pragma once


#include "cuda_runtime.h"
#include <utility>

#include "DevicePtr.h"
#include "DeviceMemory.h"


namespace cudacpp {



namespace detail_DeviceObject {

template<typename T, typename... ARGS>
__global__ void AllocateObject(DevicePtr<T*> p, ARGS... args) {
	//p[0] = static_cast<T*>(malloc(sizeof(T)));
	//new (p[0]) T(args...);
	p[0] = new T(args...);

	int a = p[0]->operator()(2, 3);
}

template<typename T>
__global__ void DeleteObject(DevicePtr<T*> p) {
	free(p[0]);
}

}

template<typename T>
class DeviceObject {
	T* _p = nullptr;
	DeviceMemory<T*> _pp;

	template<typename... ARGS>
	static void AllocateObjectOnDevice(DevicePtr<T*> pp, ARGS... args) {
		detail_DeviceObject::AllocateObject<<<1, 1>>>(pp, args...);
		auto res = cudaDeviceSynchronize();
	}

	static void DeleteObjectOnDevice(DevicePtr<T*> pp) {
		detail_DeviceObject::DeleteObject<<<1, 1>>>(pp);
		auto res = cudaDeviceSynchronize();
	}

public:
	template<typename... ARGS>
	DeviceObject(ARGS... args) 
		: _pp(DeviceMemory<T*>::AllocateElements(1))
	{
		AllocateObjectOnDevice(_pp, args...);
		auto res = CopyElements(&_p, _pp, 1);
	}

	DeviceObject(DeviceObject &&from)
		: _pp(std::move(from._pp))
	{}

	DeviceObject& operator=(DeviceObject &&from) {
		_pp = std::move(from._pp);
		return *this;
	}

	~DeviceObject() {
		if (_pp) {
			DeleteObjectOnDevice(_pp);
		}
	}

	DeviceObject(const DeviceObject&) = delete;
	DeviceObject& operator=(const DeviceObject&) = delete;

public:
	operator DevicePtr<T>() const { 
		return DevicePtr<T>::FromRawDevicePtr(_p);
	}
};


}