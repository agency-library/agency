#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Container>
struct container_factory
{
  template<class Shape>
  __AGENCY_ANNOTATION
  Container operator()(const Shape& shape) const
  {
    return Container(shape);
  }
};


} // end executor_traits_detail
} // end detail
} // end agency

