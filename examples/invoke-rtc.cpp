#ifdef FIXED_GRID_INDEX


#include "cudacpp\rtc\Program.h"
#include "cudacpp\CudaDevice.h"
#include "cudacpp\CudaContext.h"
#include "cudacpp\rtc\Kernel.h"

#include <numeric>
#include <iostream>

using namespace cudacpp;

int testinvokeRTC(int size) {
	CudaDevice dev = CudaDevice::FindByProperties(CudaDeviceProperties::ByIntegratedType(false));
	dev.setAsCurrent();
	CudaContext ctx{ dev };
 
	rtc::Program prog("myprog", rtc::Code::FromFile("..\\examples\\code-rtc.cu"));

	auto kernel = rtc::Kernel("setKernel").instantiate<float, std::integral_constant<int, 10>>();

	prog.registerKernel(kernel);


	prog.compile({
		rtc::options::GpuArchitecture(dev.properties()),
		rtc::options::CPPLang(rtc::options::CPP_x14)
	});


	rtc::Module module{ ctx, prog };
	kernel.init(module, prog);


	Grid<3, 3, 3> grid{ Size<1>(1), Size<1>(32), Size<1>(0) };

	float *ptr;
	cudaMallocManaged(&ptr, sizeof(float) * 32 * 10);
	std::fill_n(ptr, 32 * 10, 0.0f);

	auto res = kernel.launchAndWait(Stream::Default(), grid, 0, ptr, 23.1f);

	auto check = std::accumulate(ptr, ptr + 32 * 10, true, [](auto res, auto num) {
		return res && num == 23.1f;
	});

	std::cout << "testinvokeRTC: " << (check ? "PASSED" : "FAILED") << '\n';
	return 0;
}


#endif