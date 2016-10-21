#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


template<class T>
const T& min(const T& a, const T& b)
{
  return b < a ? b : a;
}

template<class T, class Compare>
const T& min(const T& a, const T& b, Compare comp)
{
  return comp(b, a) ? b : a;
}


} // end detail
} // end agency

