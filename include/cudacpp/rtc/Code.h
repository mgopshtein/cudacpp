#pragma once

#include <string>
#include <fstream>
#include <streambuf>

namespace cudacpp {
namespace rtc {


/**
 * class Code
 * ----------
 * Stores a CUDA code as a string.
 *
 * Code c{ <params> };                      // initializes the internal std::string with provided <params>
 * Code c = Code::FromFile(fname);          // loads a content of the file to the string
 * const string &code = c.code();		   // get reference to internal string
 */
class Code {
	const std::string _code;

public:
	template<typename... ARGS>
	explicit Code(ARGS&& ...args)
		: _code(std::forward<ARGS>(args)...)
	{}

	static Code FromFile(const std::string &name) {
		std::ifstream t(name);
		if (!t.good()) {
			throw std::exception("Can't read file");
		}
		std::string str;

		t.seekg(0, std::ios::end);
		str.reserve(t.tellg());
		t.seekg(0, std::ios::beg);

		str.assign(
			(std::istreambuf_iterator<char>(t)),
			std::istreambuf_iterator<char>());

		return Code{ std::move(str) };
	}

	const auto & code() const { return _code; }
};


}
}


