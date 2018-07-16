#pragma once

#include <vector>

#include "Module.h"
#include "..\Grid.h"
#include "..\Stream.h"

namespace cudacpp {
namespace rtc {

/**
 * class Kernel
 * ------------
 * Manages and runs CUDA kernel.
 *
 * Kernel k(name);                           // Constructs the Kernel object with given name
 *
 * k.instantiate<template params>();         // For template kernels, defines the template agruments for the Kernel.
 *                                           // The parameters can include a type - for typename|class, or integral_constant for integrals
 * <EXAMPLE>
 * Assume in the CUDA code you have
 *
 * template<typename DATA, int UNROLL>
 * void process(DATA *data) {...}
 *
 * Then for "process<float, 10>" you create the Kernel object as:
 *
 * Kernel k("process");
 * k.instantiate<float, std::integral_constant<int, 10>>();
 *
 * alternatively you can use TemplateParameters class directly:
 *
 * TemplateParameters tp;
 * tp.addType<float>();
 * tp.addValue(10);
 * k.instantiate(tp);
 * </EXAMPLE>
 *
 * k.launch(stream, stream, shared_mem_size, params);      // launches the kernel
 * k.launchAndWait(...same as launch...);                  // launches the kernel, and waits for it to complete
 */
class Kernel {
	CUfunction _kernel = nullptr;
	std::string _name;

public:
	class TemplateParameters;

	Kernel(const std::string &name);

	Kernel & instantiate(const TemplateParameters&);

	// builds template parameters string automatically
	template<typename... ARGS>
	Kernel & instantiate();

	const auto & getFullName() const { return _name; }

public:
 	void init(const Module &m, const Program &p) {
 		auto res = cuModuleGetFunction(&_kernel, m.module(), p.loweredName(*this).c_str());
 	}

	template<typename... ARGS>
	CUresult launch(const Stream& stream, const Grid<3, 3, 3>& grid, unsigned int sharedMem, const ARGS& ...args);

	template<typename... ARGS>
	CUresult launchAndWait(const Stream& stream, const ARGS& ...args) {
		auto res = launch(stream, args...);
		if (res != CUDA_SUCCESS) {
			return res;
		}
		return stream.synchronize();
	}
};


namespace detail {

	template<typename... ARGS>
	static inline std::vector<void*> BuildArgs(const ARGS& ...args) {
		return{ const_cast<void*>(reinterpret_cast<const void*>(&args))... };
	}


	template<typename T>
	struct NameExtractor {
		static std::string extract() {
			std::string type_name;
			nvrtcGetTypeName<T>(&type_name);
			return type_name;
		}
	};

	template<typename T, T t>
	struct NameExtractor<std::integral_constant<T, t>> {
		static std::string extract() {
			return std::to_string(t);
		}
	};
}

class Kernel::TemplateParameters {
	std::string _val;
	bool _first = true;

	void addComma() {
		if (!_first) {
			_val += ',';
		}
		else {
			_first = false;
		}
	}

public:
	template<typename T>
	auto & addValue(const T& t) {
		addComma();
		_val += std::to_string(t);
		return *this;
	}

	template<typename T>
	auto & addType() {
		addComma();
		_val += detail::NameExtractor<T>::extract();
		return *this;
	}

	const std::string & operator()() const { return _val; }
};



namespace detail {


	template<typename T, typename U, typename... REST>
	static inline auto AddTypesToTemplate(Kernel::TemplateParameters& params) {
		params.addType<T>();
		AddTypesToTemplate<U, REST...>(params);
	}

	template<typename T>
	static inline void AddTypesToTemplate(Kernel::TemplateParameters& params) {
		params.addType<T>();
	}

	static inline void AddTypesToTemplate(Kernel::TemplateParameters& params) {}

}




/// for function with no template parameters
inline Kernel::Kernel(const std::string &name)
	: _name(name)
{}

inline Kernel & Kernel::instantiate(const TemplateParameters &tp) {
	_name = _name + '<' + tp() + '>';
	return *this;
}


template<typename... ARGS>
inline Kernel & Kernel::instantiate() {
	TemplateParameters tp;
	detail::AddTypesToTemplate<ARGS...>(tp);
	return instantiate(tp);
}



template<typename... ARGS>
inline CUresult Kernel::launch(const Stream& stream, const Grid<3, 3, 3>& grid, unsigned int sharedMem, const ARGS& ...args) {
	auto vec = detail::BuildArgs(args...);
	auto vecPtr = (vec.empty()) ? nullptr : vec.data();
	return cuLaunchKernel(
		_kernel,
		grid.blocks.x, grid.blocks.y, grid.blocks.z,
		grid.threads.x, grid.threads.y, grid.threads.z,
		sharedMem,
		stream(),
		vecPtr,
		nullptr /*extra*/
	);
}

}
}
