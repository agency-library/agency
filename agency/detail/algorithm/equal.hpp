#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{


template<class InputIterator1, class InputIterator2>
__AGENCY_ANNOTATION
bool equal(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2)
{
  for(; first1 != last1; ++first1, ++first2)
  {
    if(!(*first1 == *first2))
    {
      return false;
    }
  }

  return true;
}


} // end detail
} // end agency

