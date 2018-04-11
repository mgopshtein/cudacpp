
//#include "cudacpp\DeviceVector.h"

template<typename type, int size>
__global__ void setKernel(type* c, type val)
{
	auto idx = threadIdx.x * size;

	#pragma unroll(size)
	for (auto i = 0; i < size; i++) {
		c[idx] = val;
		idx++;
	}
}