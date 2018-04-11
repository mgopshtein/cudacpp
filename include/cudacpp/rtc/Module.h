#pragma once

#include "Program.h"
#include "..\CudaContext.h"

namespace cudacpp {
namespace rtc {


class Module {
	CUmodule _module;

public:
	Module(const CudaContext&, const Program &p) {
		cuModuleLoadDataEx(&_module, p.PTX().c_str(), 0, 0, 0);
	}

	auto module() const { return _module; }
};


}
}
