#pragma once

#ifndef NVRTC_GET_TYPE_NAME
#	define NVRTC_GET_TYPE_NAME 1
#endif // !NVRTC_GET_TYPE_NAME

#include <nvrtc.h>
#include <cuda.h>

#include <type_traits>

#include "Code.h"
#include "Header.h"
#include "CompilationOptions.h"



namespace cudacpp {
namespace rtc {


class Kernel;


/**
 * class Program
 * -------------
 * Handles CUDA program.
 *
 * Code c;
 * Program p{name, c};                  // constructs a Program from existing code
 * Program p{name, c, headers};         // constructs a Program from existing code, while providing header files needed for compilation
 *
 * p.registerKernel(k);                 // this must be done before the compilation, for all the kernels having template parameters
 * p.compile(flags);                    // compiles the code using the flags
 * string n = p.loweredName(k);			// returns the decorated name of tempalte kernel
 *
 * string ptx = p.PTX();                // returns the compiled PTX code of the program
 */
class Program {
	nvrtcProgram _prog;

public:
	Program(const std::string &name, const Code	&code, const std::vector<Header> &headers);

	Program(const std::string &name, const Code	&code)
		: Program(name, code, {})
	{}

	void compile(const CompilationOptions &opt = {});

public:

	void registerKernel(const Kernel&);

	std::string loweredName(const Kernel &k) const;

public:
	Program()
		: _prog(nullptr)
	{}

	Program(const Program&) = delete;
	Program & operator=(const Program&) = delete;

	Program(Program &&from) {
		std::swap(_prog, from._prog);
	}

	Program & operator=(Program &&from) {
		std::swap(_prog, from._prog);
		return *this;
	}

	~Program() {
		if (_prog) {
			nvrtcDestroyProgram(&_prog);
		}
	}

	std::string PTX() const;

};



inline Program::Program(const std::string &name, const Code &code, const std::vector<Header> &headers) {
	// prepare the arrays for headers
	auto nh = headers.size();
	std::vector<const char *> headersContent;
	std::vector<const char *> headersNames;
	for (const Header &h : headers) {
		headersContent.push_back(h.code().c_str());
		headersNames.push_back(h.name().c_str());
	}

	// create the program
	auto createRes = nvrtcCreateProgram(
		&_prog,
		code.code().c_str(),
		name.c_str(),
		(int)nh,
		(nh > 0) ? headersContent.data() : nullptr,
		(nh > 0) ? headersNames.data() : nullptr
	);
}


inline std::string Program::PTX() const {
	std::size_t size = 0;
	nvrtcGetPTXSize(_prog, &size);
	std::string res(size, '\0');
	nvrtcGetPTX(_prog, &res.front());
	return res;
}

inline void Program::compile(const CompilationOptions &opt) {
	auto res = nvrtcCompileProgram(
		_prog,
		(int)opt.numOptions(),
		opt.options()
	);

	if (res != NVRTC_SUCCESS) {
		std::size_t logSize;
		nvrtcGetProgramLogSize(_prog, &logSize);
			
		std::string log(logSize, '\0');
		nvrtcGetProgramLog(_prog, &(log.front()));

		throw std::exception(log.c_str());
	}
}

}
}


// NOTE: including at the end to avoid cyclic include dependency
#include "Kernel.h"



namespace cudacpp {
namespace rtc {

inline void Program::registerKernel(const Kernel &k) {
	auto res = nvrtcAddNameExpression(_prog, k.getFullName().c_str());
}

inline std::string Program::loweredName(const Kernel &k) const {
	const char *lowered = "";
	auto res = nvrtcGetLoweredName(_prog, k.getFullName().c_str(), &lowered);
	return lowered;
}

}
}