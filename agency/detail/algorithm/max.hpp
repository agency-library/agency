#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


template<class T>
const T& max(const T& a, const T& b)
{
  return a < b ? b : a;
}

template<class T, class Compare>
const T& max(const T& a, const T& b, Compare comp)
{
  return comp(a, b) ? b : a;
}


} // end detail
} // end agency

