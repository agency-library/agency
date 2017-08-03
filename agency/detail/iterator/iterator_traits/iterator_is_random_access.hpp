#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
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


// XXX WAR nvbug 1965154
//template<class... Iterators>
//using iterators_are_random_access = conjunction<
//  iterator_is_random_access<Iterators>...
//>;
template<class... Iterators>
struct iterators_are_random_access : conjunction<
  iterator_is_random_access<Iterators>...
>
{};


} // end detail
} // end agency

