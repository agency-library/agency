#pragma once

#include <agency/detail/config.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


template<class RandomAccessIterator, class Distance>
__AGENCY_ANNOTATION
void advance(std::random_access_iterator_tag, RandomAccessIterator& iter, Distance n)
{
  iter += n;
}

template<class InputIterator, class Distance>
__AGENCY_ANNOTATION
void advance(std::input_iterator_tag, InputIterator& iter, Distance n)
{
  for(Distance i = 0; i != n; ++i)
  {
    ++iter;
  }
}

template<class InputIterator, class Distance>
__AGENCY_ANNOTATION
void advance(InputIterator& iter, Distance n)
{
  detail::advance(typename std::iterator_traits<InputIterator>::iterator_category(), iter, n);
}


} // end detail
} // end agency

