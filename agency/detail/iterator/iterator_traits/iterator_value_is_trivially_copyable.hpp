#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Iterator>
using iterator_value_is_trivially_copyable = std::is_trivially_copyable<
  typename std::iterator_traits<Iterator>::value_type
>;


template<class... Iterators>
using iterator_values_are_trivially_copyable = conjunction<
  iterator_value_is_trivially_copyable<Iterators>...
>;


} // end detail
} // end agency

