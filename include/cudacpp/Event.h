#pragma once


#include "cuda_runtime.h"
#include "cuda.h"

#include <chrono>

namespace cudacpp {


class Event
{
	CUevent _event;
	friend class Stream;

public:
	Event() {
		cuEventCreate(&_event, CU_EVENT_DEFAULT /*CU_EVENT_BLOCKING_SYNC*/);
	}

	~Event() {
		cuEventDestroy(_event);
	}

	Event(const Event&) = delete;
	Event& operator=(const Event&) = delete;

	auto wait() {
		return cuEventSynchronize(_event);
	}

	auto operator-(const Event& e) {
		float msec;
		cuEventElapsedTime(&msec, e._event, _event);
		return std::chrono::microseconds((int)(msec * 1000.f));
	}

};	


static inline std::chrono::microseconds operator-(const Event& to, const Event& from) {
	return to - from;
}

}

