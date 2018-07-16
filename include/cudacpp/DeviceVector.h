#pragma once

#include "cuda_runtime.h"
#include <utility>

#include "DevicePtr.h"
#include "Size.h"
#include "Index.h"

namespace cudacpp {



template<typename T>
class DeviceVector {
	DevicePtr<T> _p;
	Size<1> _sz;

public:
	constexpr static int DimIndex = 1;
	using Index = Index<DimIndex>;
	using DataT = T;

	DeviceVector(DevicePtr<T> p, DimSize sz)
		: _p(p)
		, _sz(sz)
	{}

	DeviceVector() = delete;

	__DevHostI__ T& operator[](std::size_t i) { return _p[i]; }
	__DevHostI__ T& operator[](Index i) { return _p[i.x]; }
	__DevHostI__ const T& operator[](std::size_t i) const { return _p[i]; }
	__DevHostI__ const T& operator[](Index i) const { return _p[i.x]; }
	__DevHostI__ operator T* () { return _p; }
	__DevHostI__ auto size() { return _sz; }
};


}

