#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/iterator/constant_iterator.hpp>
#include <agency/experimental/ranges/counted.hpp>


namespace agency
{
namespace experimental
{


template<class T, class Difference>
__AGENCY_ANNOTATION
counted_view<agency::detail::constant_iterator<T>, Difference>
  repeat(const T& value, const Difference& n)
{
  agency::detail::constant_iterator<T> iter(value);
  return {iter, n};
}


} // end experimental
} // end agency

