#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


template<class InputIterator, class OutputIterator>
__AGENCY_ANNOTATION
OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
  for(; first != last; ++first, ++result)
  {
    *result = *first;
  }

  return result;
}


} // end detail
} // end agency

