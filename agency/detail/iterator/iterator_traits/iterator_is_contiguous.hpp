#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// XXX for the moment, only consider pointers to be contiguous iterators
template<class Iterator>
using iterator_is_contiguous = std::is_pointer<Iterator>;


template<class... Iterators>
using iterators_are_contiguous = conjunction<
  iterator_is_contiguous<Iterators>...
>;


} // end detail
} // end agency

