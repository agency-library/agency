#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class T, class Shape>
struct single_element_container
{
  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  single_element_container() : element{} {}

  __agency_exec_check_disable__
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

