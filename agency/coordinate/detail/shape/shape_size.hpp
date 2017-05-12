#pragma once

#include <agency/detail/config.hpp>
#include <tuple>
#include <type_traits>

namespace agency
{
namespace detail
{

// shape_size<i,Shape> is a type trait which returns the
// number of elements in a Shape.
//
// A Shape is either:
// 1. An integral type or
// 2. A tuple of Shapes
//
// shape_element's implementation below needs to handle these two cases.


// case 2: Shape is a Tuple-like type
template<class Shape, class Enable = void>
struct shape_size_impl : std::tuple_size<Shape> {};


// case 1: Shape is an integral type.
template<class Shape>
struct shape_size_impl<
  Shape,
  typename std::enable_if<
    std::is_integral<Shape>::value
  >::type
> : std::integral_constant<std::size_t, 1>
{
};


template<class Shape>
struct shape_size : std::integral_constant<size_t, shape_size_impl<Shape>::value> {};


} // end detail
} // end agency

