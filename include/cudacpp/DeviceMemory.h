#pragma once

#include "cuda_runtime.h"
#include <utility>

#include "DevicePtr.h"
#include "DeviceVector.h"


namespace cudacpp {


template<typename T>
class DeviceMemory {
	T *_p = nullptr;
	std::size_t _bytes = 0;

	DeviceMemory(std::size_t bytes)
		: _bytes(bytes)
	{
		cudaMalloc(&_p, _bytes);
	}

	void release() {
		if (_p) {
			cudaFree(_p);
		}
	}

public:
	DeviceMemory() = default;

	static DeviceMemory AllocateBytes(std::size_t bytes) {
		return { bytes };
	}

	static DeviceMemory AllocateElements(std::size_t els) {
		return { els * sizeof(T) };
	}

	DeviceMemory(DeviceMemory &&from)
		: _p(from._p)
		, _bytes(from._bytes)
	{
		from._p = nullptr;
	}

	DeviceMemory& operator=(DeviceMemory &&from) {
		std::swap(_p,     from._p);
		std::swap(_bytes, from._bytes);
		return *this;
	}

	~DeviceMemory() {
		release();
	}

	DeviceMemory(const DeviceMemory&) = delete;
	DeviceMemory& operator=(const DeviceMemory&) = delete;

	auto sizeBytes() const { return _bytes; }
	auto sizeElements() const { return sizeBytes() / sizeof(T); }

public:
	operator bool() const {
		return _p != nullptr;
	}

	operator DevicePtr<T>() const { 
		return DevicePtr<T>::FromRawDevicePtr(_p);
	}

	operator DeviceVector<T>() const {
		return DeviceVector<T>(DevicePtr<T>::FromRawDevicePtr(_p), _bytes / sizeof(T));
	}
};


template<typename T>
auto CopyElements(T *to, DevicePtr<const T> from, std::size_t num) {
	return cudaMemcpy(to, from, num * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
auto CopyElements(DevicePtr<T> to, const T *from, std::size_t num) {
	return cudaMemcpy(to, from, num * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
auto CopyElements(T *to, DeviceMemory<T> &from, std::size_t num) {
	return CopyElements<T>(to, (DevicePtr<T>) from, num);
}

template<typename T>
auto CopyElements(DeviceMemory<T> &to, const T *from, std::size_t num) {
	return CopyElements<T>((DevicePtr<T>)to, from, num);
}

}


