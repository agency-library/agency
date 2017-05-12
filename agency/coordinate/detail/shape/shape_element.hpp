#pragma once

#include <agency/detail/config.hpp>
#include <tuple>
#include <type_traits>

namespace agency
{
namespace detail
{


// shape_element<i,Shape> is a type trait which returns the
// type of the ith element of the given Shape.
//
// A Shape is either:
// 1. An integral type or
// 2. A tuple of Shapes
//
// shape_element's implementation below needs to handle these two cases.


// case 2: Shape is a Tuple-like type
template<size_t i, class Shape, class Enable = void>
struct shape_element_impl : std::tuple_element<i, Shape> {};


// case 1: Shape is an integral type.
// This case only makes sense when i is 0.
template<size_t i, class Shape>
struct shape_element_impl<
  i,
  Shape,
  typename std::enable_if<
    std::is_integral<Shape>::value && i == 0
  >::type
>
{
  using type = Shape;
};


template<size_t i, class Shape>
struct shape_element
{
  using type = typename shape_element_impl<i,Shape>::type;
};

template<size_t i, class Shape>
using shape_element_t = typename shape_element<i,Shape>::type;


} // end detail
} // end agency

