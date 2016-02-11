#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class T>
struct single_element_container
{
  __AGENCY_ANNOTATION
  single_element_container() {}

  __agency_hd_warning_disable__
  template<class Shape>
  __AGENCY_ANNOTATION
  single_element_container(const Shape&) : element{} {}

  template<class Index>
  __AGENCY_ANNOTATION
  T& operator[](const Index&)
  {
    return element;
  }

  T element;
};


} // end executor_traits_detail
} // end detail
} // end agency

