#pragma once

#include <utility>

#include "cuda_runtime.h"
#include "cuda.h"

#include "Size.h"
#include "Event.h"

namespace cudacpp {


class Stream {
	CUstream _stream;

	Stream(CUstream stream) : _stream(stream) {}

public:
	enum Flags {
		Stream_Default,
		Stream_NonBlocking		// doesn't block with the default 0 stream
	};

public:
	// creates the default stream
	Stream(Flags flags = Stream_Default) {
		auto res = cuStreamCreate(
			&_stream,
			(flags == Stream_Default) ? CU_STREAM_DEFAULT : CU_STREAM_NON_BLOCKING
		);
	}

	Stream(const Stream&) = delete;
	Stream& operator=(const Stream&) = delete;

	Stream(Stream &&from) : _stream(from._stream) { from._stream = nullptr; }
	Stream& operator=(Stream &&from) { std::swap(_stream, from._stream); return *this; }

	~Stream() {
		if (_stream) {
			cuStreamDestroy(_stream);
		}
	}

	static Stream Default() { return Stream((CUstream)nullptr); }

	CUstream operator()() const { return _stream; }
	auto synchronize() const { return cuStreamSynchronize(_stream); }

	void record(Event& e) {
		cuEventRecord(e._event, _stream);
	}
};

}
