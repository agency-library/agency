#pragma once

#include <agency/detail/config.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


template<class RandomAccessIterator>
__AGENCY_ANNOTATION
typename std::iterator_traits<RandomAccessIterator>::difference_type distance(std::random_access_iterator_tag, RandomAccessIterator first, RandomAccessIterator last)
{
  return last - first;
}

template<class InputIterator>
__AGENCY_ANNOTATION
typename std::iterator_traits<InputIterator>::difference_type distance(std::input_iterator_tag, InputIterator first, InputIterator last)
{
  typename std::iterator_traits<InputIterator>::difference_type result = 0;

  for(; first != last; ++first)
  {
    ++result;
  }

  return result;
}

template<class InputIterator>
__AGENCY_ANNOTATION
typename std::iterator_traits<InputIterator>::difference_type distance(InputIterator first, InputIterator last)
{
  return detail::distance(typename std::iterator_traits<InputIterator>::iterator_category(), first, last);
}


} // end detail
} // end agency

