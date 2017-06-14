#pragma once

#include <agency/detail/config.hpp>
#include <iterator>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Iterator>
using iterator_is_random_access = std::is_convertible<
  typename std::iterator_traits<Iterator>::iterator_category,
  std::random_access_iterator_tag
>;


} // end detail
} // end agency


