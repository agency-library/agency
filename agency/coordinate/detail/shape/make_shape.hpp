#pragma once

#include <agency/detail/config.hpp>
#include <utility>
#include <array>

namespace agency
{
namespace detail
{


template<class Shape>
struct make_shape_impl
{
  template<class... Args>
  __AGENCY_ANNOTATION
  static Shape make(Args&&... args)
  {
    return Shape{std::forward<Args>(args)...};
  }
};

// specialization for std::array, which requires the weird doubly-nested brace syntax
template<class T, size_t n>
struct make_shape_impl<std::array<T,n>>
{
  template<class... Args>
  __AGENCY_ANNOTATION
  static std::array<T,n> make(Args&&... args)
  {
    return std::array<T,n>{{std::forward<Args>(args)...}};
  }
};


// make_shape makes a Shape from a list of elements
// XXX should probably require that the number of Args... matches shape_size
template<class Shape, class... Args>
__AGENCY_ANNOTATION
Shape make_shape(Args&&... args)
{
  return make_shape_impl<Shape>::make(std::forward<Args>(args)...);
}


} // end detail
} // end agency

