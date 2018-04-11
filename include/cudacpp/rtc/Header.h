#pragma once


#include "Code.h"

namespace cudacpp {
namespace rtc {

/**
 * class Header
 *    : public Code
 * ------------
 * Stores a CUDA code and the include name of a header file.
 *
 * Header h{ name, <params> };              // initializes the include name and the Code content with provided <params>
 * const string &name = h.name();		    // get the include name
 * [see also Code class public members]
 */
class Header 
	: public Code
{
	const std::string _name;

public:
	template<typename... ARGS>
	Header(const std::string &name, ARGS&& ...args)
		: Code(std::forward<ARGS>(args)...)
		, _name(name)
	{}

	const auto & name() const { return _name; }
};





}
}
