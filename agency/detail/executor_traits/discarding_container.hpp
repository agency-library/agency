#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// this class provides the facade of a container type
// but discards assignments to its elements
struct discarding_container
{
  struct reference
  {
    template<class T>
    __AGENCY_ANNOTATION
    reference& operator=(const T&) { return *this; }
  };

  template<class... Args>
  __AGENCY_ANNOTATION
  discarding_container(Args&&...) {}

  template<class Index>
  __AGENCY_ANNOTATION
  reference operator[](const Index&) const
  {
    return reference();
  }
};


} // end new_executor_traits_detail
} // end detail
} // end agency

