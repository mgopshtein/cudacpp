#pragma once

#include "cuda_runtime.h"
#include <utility>
#include <type_traits>

#include "Size.h"
#include "Defs.h"

namespace cudacpp {



template<typename T>
class DevicePtr {
	T *_p = nullptr;

	__DevHostI__ DevicePtr(T *p) : _p(p) {}

public:
	__host__ static DevicePtr FromRawDevicePtr(T *p) {
		return { p };
	}

	template< typename T1, typename = std::enable_if_t<std::is_convertible_v<T1*, T*>> >
	__DevHostI__ DevicePtr(const DevicePtr<T1> &dp)
		: _p((T1*)dp)
	{}

public:
	__DevI__ T* operator->() const { return  _p; }
	__DevI__ T& operator*()  const { return *_p; }
	__DevHostI__ operator T*()   const { return  _p; }

	__DevI__ T& operator[](std::size_t s) { return _p[s]; }

	friend __DevI__ DevicePtr operator+(const DevicePtr &dev, std::size_t s) {
		return { dev._p + s };
	}

	__DevI__ auto& operator+=(std::size_t s) {
		_p += s;
		return *this;
	}

	__DevI__ auto& operator++() {
		_p++;
		return *this;
	}

	__DevI__ auto operator++(int) {
		return DevicePtr(_p + 1);
	}
};


template<typename T>
__host__ inline auto MakeDevicePtr(T* p) {
	return DevicePtr<T>::FromRawDevicePtr(p);
}


}
