#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/coordinate/detail/shape/shape_size.hpp>
#include <agency/coordinate/detail/shape/shape_element.hpp>
#include <agency/coordinate/detail/shape/shape_append.hpp>
#include <agency/tuple.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace common_shape_detail
{


template<class T1, class T2>
using common_type_t = typename std::common_type<T1,T2>::type;


// because the implementation of common_shape2 is recursive, we introduce a forward declaration so it may call itself.
template<class Shape1, class Shape2>
struct common_shape2;

template<class Shape1, class Shape2>
using common_shape2_t = typename common_shape2<Shape1,Shape2>::type;


// in the following, we implement the type trait common_shape2 using constexpr functions so that we can use __AGENCY_REQUIRES easily

// First, if the two Shapes are identical (case 0), there is nothing to do.
//
// To create a Shape type which could accomodate either of two other Shape types A or B, the idea is to create a
// Shape "wide" enough to hold either A or B.
// 
// We do this by appending dimensions to the narrower of the two Shapes (cases 1 & 2). 
// When both Shapes have the same number of dimensions (case 3), we recurse into the elements of those dimensions, which are themselves Shapes.
// We terminate the recursion (at case 4) when we compare two scalars to each other.

// case 0: The two Shapes are identical
template<class Shape>
constexpr Shape common_shape_impl(Shape, Shape)
{
  return Shape();
}

// case 1: Shape1 has fewer elements than Shape2
template<class Shape1, class Shape2,
         __AGENCY_REQUIRES(
           shape_size<Shape1>::value < shape_size<Shape2>::value
         )>
constexpr
common_shape2_t<
  shape_append_t<Shape1, shape_element_t<shape_size<Shape1>::value, Shape2>>,
  Shape2
>
  common_shape_impl(Shape1, Shape2)
{
  // append a dimension to the narrower of the two shapes, Shape1
  using type_to_append = shape_element_t<shape_size<Shape1>::value, Shape2>;
  using widened_shape = shape_append_t<Shape1, type_to_append>;

  // recurse
  return common_shape2_t<widened_shape, Shape2>();
};


// case 2: Shape2 has fewer elements than Shape1
template<class Shape1, class Shape2,
         __AGENCY_REQUIRES(
           shape_size<Shape1>::value > shape_size<Shape2>::value
         )>
constexpr
common_shape2_t<
  Shape1,
  shape_append_t<Shape2, shape_element_t<shape_size<Shape2>::value, Shape1>>
>
  common_shape_impl(Shape1, Shape2)
{
  // append a dimension to the narrower of the two shapes, Shape2
  using type_to_append = shape_element_t<shape_size<Shape2>::value, Shape1>;
  using widened_shape = shape_append_t<Shape2, type_to_append>;

  // recurse
  return common_shape2_t<Shape1, widened_shape>();
}


// case 3: Shape1 and Shape2 have the same number of elements, and this number is greater than one
template<class Shape1, class Shape2,
         class Indices>
struct case_3;

template<class Shape1, class Shape2, size_t... Indices>
struct case_3<Shape1, Shape2, index_sequence<Indices...>>
{
  // when it's possible to tuple_rebind Shape1, do this to produce the resulting Shape type
  // otherwise, use detail::tuple as the resulting tuple type
  // XXX we might prefer to use something more specific than detail::tuple, e.g detail::shape_tuple
  using type = detail::tuple_rebind_if_t<
    Shape1,
    tuple,
    common_shape2_t<                       // the resulting Shape is composed of the the common shape of Shape1's & Shape2's constituent elements
      shape_element_t<Indices,Shape1>,
      shape_element_t<Indices,Shape2>
    >...
  >;
};

template<class Shape1, class Shape2>
using case_3_t = typename case_3<Shape1, Shape2, make_index_sequence<shape_size<Shape1>::value>>::type;

template<class Shape1, class Shape2,
         __AGENCY_REQUIRES(
           shape_size<Shape1>::value != 1 &&
           shape_size<Shape2>::value != 1
         ),
         __AGENCY_REQUIRES(
           shape_size<Shape1>::value == shape_size<Shape2>::value
         )>
constexpr case_3_t<Shape1,Shape2>
  common_shape_impl(Shape1, Shape2)
{
  return case_3_t<Shape1,Shape2>();
}


// case 4: both Shape1 & Shape2 both have a single element and those elements' types are integral
template<class Shape1, class Shape2,
         __AGENCY_REQUIRES(
           shape_size<Shape1>::value == 1 &&
           shape_size<Shape2>::value == 1
         ),
         __AGENCY_REQUIRES(
           std::is_integral<shape_element_t<0,Shape1>>::value &&
           std::is_integral<shape_element_t<0,Shape2>>::value
         )>
constexpr common_type_t<shape_element_t<0,Shape1>,shape_element_t<0,Shape2>>
  common_shape_impl(Shape1, Shape2)
{
  using element1 = shape_element_t<0,Shape1>;
  using element2 = shape_element_t<0,Shape2>;

  // return the common type of these two scalars
  return common_type_t<element1,element2>();
}


template<class Shape1, class Shape2>
struct common_shape2
{
  using type = decltype(common_shape_detail::common_shape_impl(std::declval<Shape1>(), std::declval<Shape2>()));
};


} // end common_shape_detail


// common_shape is a type trait which, given one or more possibly different Shapes, returns
// a Shape with dimensions sufficient to represent any of the Shapes.
template<class Shape, class... Shapes>
struct common_shape;

template<class Shape, class... Shapes>
using common_shape_t = typename common_shape<Shape,Shapes...>::type;


// the implementation of common_shape is recursive
// this is the recursive case
template<class Shape1, class Shape2, class... Shapes>
struct common_shape<Shape1, Shape2, Shapes...>
{
  using type = common_shape_t<
    Shape1,
    common_shape_t<Shape2, Shapes...>
  >;
};

// base case 1: a single Shape
template<class Shape>
struct common_shape<Shape>
{
  using type = Shape;
};

// base case 2: two Shapes
template<class Shape1, class Shape2>
struct common_shape<Shape1, Shape2>
{
  // with two Shapes, we lower onto the two Shape implementation inside common_shape_detail
  using type = common_shape_detail::common_shape2_t<Shape1,Shape2>;
};


} // end detail
} // end agency

