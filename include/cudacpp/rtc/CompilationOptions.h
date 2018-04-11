#pragma once

#include <vector>
#include <string>
#include <algorithm>

#include "..\CudaDevice.h"

namespace cudacpp {
namespace rtc {


/**
 * class CompilationOptions
 * ------------------------
 * Manages CUDA-->PFX compiler options.
 *
 * CompilationOptions co;                 // constructs empty options set
 *
 * CompilationOptions co {                // constructs options with given parameter set
 *    options::GpuArchitecture(5, 0),
 *    options::CPPLang(options::CPP_x14)
 * };
 *
 * size_t n = co.numOptions();            // returns the number of option flags defined
 * const char** opts = co.options();      // returns an array of the options in the string format
 *                                        // NOTE: the array is managed by CompilationOptions class and
 *                                        // is valid until the next time you call this function
 *
 * namespace options {                    // specific compilation options
 *    GpuArchitecture(major, minor)       // --gpu-architecture=compute_<major><minor>
 *    CPPLang(CPP_x11|CPP_x14)            // --std=c++11|c++14
 *    FMAD(true|false)                    // --fmad=true|false
 * }
 */
class CompilationOptions {
	std::vector<std::string> _options;
	mutable std::vector<const char*> _chOptions;

public:
	void insert(const std::string &op) {
		_options.push_back(op);
	}

	void insert(const std::string &name, const std::string &value) {
		if (value.empty()) {
			insert(name);
		}
		else {
			_options.push_back(name + '=' + value);
		}
	}

	template<typename T>
	void insertOptions(const T& t) {
		insert(t.name(), t.value());
	}

	template<typename T, typename... TS>
	void insertOptions(const T& t, const TS& ...ts) {
		insert(t.name(), t.value());
		insertOptions(ts...);
	}

public:
	template<typename... TS>
	CompilationOptions(TS&& ...ts)
	{
		insertOptions(ts...);
	}

	CompilationOptions() = default;

public:
	const char** options() const {
		_chOptions.resize(_options.size());
		std::transform(_options.begin(), _options.end(), _chOptions.begin(), [](const auto &s){return s.c_str();});
		return _chOptions.data();
	}

	auto numOptions() const { return _options.size(); }
};


namespace options {

class GpuArchitecture {
	const std::string _arc;

public:
	GpuArchitecture(int major, int minor) 
		: _arc(std::string("compute_") + std::to_string(major) + std::to_string(minor))
	{}

	GpuArchitecture(const CudaDeviceProperties& props)
		: GpuArchitecture(props.major(), props.minor())
	{}

	auto name() const { return "--gpu-architecture"; }
	auto & value() const { return _arc; }
};

enum CPPLangVer {
	CPP_x11,
	CPP_x14
};

class CPPLang {
	const CPPLangVer _ver;

public:
	CPPLang(CPPLangVer ver) : _ver(ver) {}

	auto name() const { return "--std"; }
	auto value() const { return (_ver == CPP_x11) ? "c++11" : "c++14"; }
};


namespace detail {
class BooleanOption {
	const bool _b;

public:
	BooleanOption(bool b) : _b(b) {}
	auto value() const { return (_b) ? "true" : "false"; }
};
}


class FMAD
	: public detail::BooleanOption
{
public:
	using detail::BooleanOption::BooleanOption;
	auto name() const { return "--fmad"; }
};

}




}
}
