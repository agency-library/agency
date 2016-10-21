#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
{


template<class InputIterator, class Size, class OutputIterator>
__AGENCY_ANNOTATION
tuple<InputIterator,OutputIterator> copy_n(InputIterator first, Size n, OutputIterator result)
{
  for(Size i = 0; i < n; ++i, ++first, ++result)
  {
    *result = *first;
  }

  return detail::make_tuple(first,result);
}


} // end detail
} // end agency

