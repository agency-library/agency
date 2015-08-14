#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T>
struct single_element_container
{
  __AGENCY_ANNOTATION
  single_element_container() {}

  template<class Shape>
  __AGENCY_ANNOTATION
  single_element_container(const Shape&)
  {
  }

  template<class Index>
  __AGENCY_ANNOTATION
  T& operator[](const Index&)
  {
    return element;
  }

  T element;
};


} // end new_executor_traits_detail
} // end detail
} // end agency

