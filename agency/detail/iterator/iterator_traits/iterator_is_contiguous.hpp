#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// XXX for the moment, only consider pointers to be contiguous iterators
template<class Iterator>
using iterator_is_contiguous = std::is_pointer<Iterator>;


} // end detail
} // end agency

