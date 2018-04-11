#pragma once


#include "CudaDevice.h"

namespace cudacpp {

class CudaContext {
	CUcontext ctx;

public:
	CudaContext(const CudaDevice& device)
		: ctx(nullptr)
	{
		cuInit(0);
		cuCtxCreate(&ctx, 0, device.handle());
	}

	CudaContext(const CudaContext&) = delete;
	CudaContext& operator=(const CudaContext&) = delete;

	~CudaContext() {
		if (ctx) {
			cuCtxDestroy(ctx);
		}
	}

};


}