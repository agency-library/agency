#pragma once

#include <agency/detail/config.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class T>
__AGENCY_ANNOTATION
void swap(T& a, T& b)
{
  T tmp = std::move(a);
  a = std::move(b);
  b = std::move(tmp);
}


template<class T>
__AGENCY_ANNOTATION
T&& decay_copy(T&& arg)
{
  return std::forward<T>(arg);
}


} // end detail
} // end agency


