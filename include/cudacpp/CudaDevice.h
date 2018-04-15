#pragma once

#include "cuda_runtime.h"
#include "cuda.h"

#include <exception>
#include <string>
#include <algorithm>
#include <vector>

namespace cudacpp {


class CudaDeviceProperties {
	cudaDeviceProp _props;

	explicit CudaDeviceProperties(const cudaDeviceProp &props) : _props(props) {}

public:
	CudaDeviceProperties(int device)
	{
		cudaGetDeviceProperties(&_props, device);

		// just in case...
		auto nameSize = sizeof(_props.name) / sizeof(_props.name[0]);
		_props.name[nameSize - 1] = '\0';
	}

	static CudaDeviceProperties FromExistingProperties(const cudaDeviceProp &props) {
		return CudaDeviceProperties{props};
	}

	static CudaDeviceProperties ByIntegratedType(bool integrated) {
		cudaDeviceProp props = { 0 };
		props.integrated = (integrated) ? 1 : 0;
		return FromExistingProperties(props);
	}

	const auto & getRawStruct() const { return _props; }
	auto major() const { return _props.major; }
	auto minor() const { return _props.minor; }
	bool integrated() const { return _props.integrated > 0; }
	const char * name() const { return _props.name; }
};


class CudaDevice {
	int _device;
	CudaDeviceProperties _props;

public:
	explicit CudaDevice(int device)
		: _device(device)
		, _props(_device)
	{}

	CUdevice handle() const;

	static CudaDevice FindByProperties(const CudaDeviceProperties&);
	static CudaDevice FindByName(std::string name);
	static CudaDevice CurrentDevice();
	static std::vector<CudaDevice> EnumerateDevices();

	static int NumberOfDevices();

	const auto & properties() const { return _props; }
	const char * name() const { return properties().name(); }

	void setAsCurrent() { cudaSetDevice(_device); }
};


inline CUdevice CudaDevice::handle() const {
	CUdevice h;
	if (CUDA_SUCCESS != cuDeviceGet(&h, _device)) {
		throw std::exception("Could not get device handle");
	}
	return h;
}


inline CudaDevice CudaDevice::FindByProperties(const CudaDeviceProperties& props) {
	int device;
	auto res = cudaChooseDevice(&device, &props.getRawStruct());
	if (res != cudaSuccess) {
		throw std::exception("Failed to find CUDA device by properties");
	}

	return CudaDevice{ device };
}


inline int CudaDevice::NumberOfDevices() {
	int numDevices = 0;
	if (cudaSuccess != cudaGetDeviceCount(&numDevices)) {
		throw std::exception("Failed to get number of CUDA devices");
	}
	return numDevices;
}


inline CudaDevice CudaDevice::FindByName(std::string name) {
	int numDevices = NumberOfDevices();
	if (numDevices == 0) {
		throw std::exception("No CUDA devices found");
	}

	std::transform(name.begin(), name.end(), name.begin(), ::tolower);
	for (int i = 0; i < numDevices; i++) {
		CudaDevice devi{ i };
		std::string deviName{ devi.name() };
		std::transform(deviName.begin(), deviName.end(), deviName.begin(), ::tolower);
		if (std::string::npos != deviName.find(name)) {
			// found by name!
			return devi;
		}
	}

	throw std::exception("Could not find CUDA device by name");
}

inline std::vector<CudaDevice> CudaDevice::EnumerateDevices() {
	std::vector<CudaDevice> res;
	int numDevices = NumberOfDevices();
	for (int i = 0; i < numDevices; i++) {
		res.emplace_back(i);
	}
	return res;
}

inline CudaDevice CudaDevice::CurrentDevice() {
	int device;
	if (cudaSuccess != cudaGetDevice(&device)) {
		throw std::exception("Can't get current device index");
	}
	return CudaDevice{ device };
}


}
