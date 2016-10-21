#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class T>
__AGENCY_ANNOTATION
const T& min(const T& a, const T& b)
{
  return b < a ? b : a;
}


__agency_exec_check_disable__
template<class T, class Compare>
__AGENCY_ANNOTATION
const T& min(const T& a, const T& b, Compare comp)
{
  return comp(b, a) ? b : a;
}


} // end detail
} // end agency

