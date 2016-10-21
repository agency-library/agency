#pragma once

#include <agency/detail/config.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


template<class RandomAccessIterator, class Distance>
__AGENCY_ANNOTATION
RandomAccessIterator& advance(std::random_access_iterator_tag, RandomAccessIterator& iter, Distance n)
{
  iter += n;
  return iter;
}

template<class InputIterator, class Distance>
__AGENCY_ANNOTATION
InputIterator& advance(std::input_iterator_tag, InputIterator& iter, Distance n)
{
  for(Distance i = 0; i != n; ++i)
  {
    ++iter;
  }

  return iter;
}

template<class InputIterator, class Distance>
__AGENCY_ANNOTATION
InputIterator& advance(InputIterator& iter, Distance n)
{
  return detail::advance(typename std::iterator_traits<InputIterator>::iterator_category(), iter, n);
}


} // end detail
} // end agency

