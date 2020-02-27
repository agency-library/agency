#pragma once

#include <type_traits>
#include <tuple>
#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/tuple.hpp>
#include <agency/coordinate.hpp>
#include <agency/coordinate/detail/shape/shape_cast.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <agency/detail/point_size.hpp>

namespace agency
{
namespace detail
{


template<class Tuple,
         class = typename std::enable_if<
           !std::is_integral<
              typename std::decay<Tuple>::type
            >::value
         >::type>
__AGENCY_ANNOTATION
auto wrap_scalar(Tuple&& t)
  -> decltype(std::forward<Tuple>(t))
{
  return std::forward<Tuple>(t);
}


template<class T,
         class = typename std::enable_if<
           std::is_integral<
             typename std::decay<T>::type
           >::value
         >::type>
__AGENCY_ANNOTATION
auto wrap_scalar(T&& x)
  -> decltype(
       rebind_point_size_t<
         typename std::decay<T>::type,
         1
       >{std::forward<T>(x)}
     )
{
  return rebind_point_size_t<
    typename std::decay<T>::type,
    1
  >{std::forward<T>(x)};
}


template<int Dimension, class IndexTuple, class ShapeTuple, class Size,
         __AGENCY_REQUIRES(
           Dimension == 0
         )>
__AGENCY_ANNOTATION
void increment_index(IndexTuple& idx, const ShapeTuple&, Size sz)
{
  agency::get<Dimension>(idx) += sz;
}


template<int Dimension, class IndexTuple, class ShapeTuple, class Size,
         __AGENCY_REQUIRES(
           Dimension > 0
         )>
__AGENCY_ANNOTATION
void increment_index(IndexTuple& idx, const ShapeTuple& shape, Size sz)
{
  Size size_of_current_dimension = agency::get<Dimension>(shape);

  while(agency::get<Dimension>(idx) + sz >= size_of_current_dimension)
  {
    // handle a carry: increment the dimension to the left
    increment_index<Dimension-1>(idx, shape, 1);

    // decrement by the size of the current dimension
    sz -= size_of_current_dimension;
  }

  agency::get<Dimension>(idx) += sz;
}


// project_index drops the highest dimension from idx & shape
// while preserving the number of points in the lower-dimensional shape
// and also preserving the rank of the index within the index space
template<class IndexTuple, class ShapeTuple>
__AGENCY_ANNOTATION
auto project_index(const IndexTuple& idx, const ShapeTuple& shape)
  -> typename std::decay<
       decltype(
         detail::unwrap_single_element_tuple(detail::tuple_drop_last(idx))
    )>::type
{
  // find the size of the dimension we'll be dropping
  auto size_of_last_dimension = detail::index_space_size(detail::tuple_last(shape));

  // lower the dimension of the shape
  auto lower_dimensional_shape = detail::project_shape(shape);

  // drop the last dimension of the index
  auto lower_dimensional_idx = detail::tuple_drop_last(idx);

  // we will increment the lower dimensional index by the amount we dropped
  auto remainder = detail::tuple_last(idx);

  // count the number of dimensions in the result
  constexpr size_t num_dimensions = std::tuple_size<decltype(lower_dimensional_idx)>::value;

  // multiply the last dimension in the output by the size of the dimension we dropped from the input
  detail::tuple_last(lower_dimensional_idx) *= size_of_last_dimension;

  // increment the output by the remainder
  detail::increment_index<num_dimensions - 1>(lower_dimensional_idx, lower_dimensional_shape, remainder);

  // if we resulted in a single-element tuple, then unwrap it before returning
  return detail::unwrap_single_element_tuple(lower_dimensional_idx);
}


namespace index_cast_detail
{


// to lift a point-like type, add a dimension
template<class Index>
struct lift_t_impl
{
  using type = rebind_point_size_t<Index, point_size<Index>::value + 1>;
};


// to lift a heterogeneous Tuple-like type, prepend the type of the first element
template<template<class...> class tuple, class T, class... Types>
struct lift_t_impl<tuple<T,Types...>>
{
  using type = tuple<T,T,Types...>;
};


template<class T>
struct make
{
  template<class... Args>
  __AGENCY_ANNOTATION
  T operator()(Args&&... args) const
  {
    return T{std::forward<Args>(args)...};
  }
};


} // end index_cast_detail


template<class Point>
using lift_t = typename index_cast_detail::lift_t_impl<Point>::type;


template<class Index, class FromShape, class ToShape>
__AGENCY_ANNOTATION
agency::tuple<lift_t<Index>, lift_t<FromShape>>
  lift_index(const Index& idx, const FromShape& from_shape, const ToShape& to_shape)
{
  // ensure that the input is a tuple so that we can use agency::get
  auto idx_tuple = wrap_scalar(idx);

  auto intermediate_result = idx_tuple;

  // mod the leading dimension of the input by the corresponding dimension in the target index space
  constexpr size_t corresponding_dim_in_to_shape = std::tuple_size<ToShape>::value - std::tuple_size<decltype(intermediate_result)>::value;

  agency::get<0>(intermediate_result) %= agency::get<corresponding_dim_in_to_shape>(to_shape);

  auto index_maker = index_cast_detail::make<lift_t<Index>>{};

  auto lifted_index = __tu::tuple_prepend_invoke(intermediate_result, agency::get<0>(idx_tuple) / agency::get<corresponding_dim_in_to_shape>(to_shape), index_maker);

  // to lift from_shape, simply append the element of to_shape we just divided by
  auto shape_maker = index_cast_detail::make<lift_t<FromShape>>{};
  auto lifted_shape = __tu::tuple_prepend_invoke(wrap_scalar(from_shape), agency::get<corresponding_dim_in_to_shape>(to_shape), shape_maker);

  return agency::make_tuple(lifted_index, lifted_shape);
}


// index_cast is recursive and has various overloads
// declare them here before their definitions

// scalar -> scalar (base case)
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value == index_size<ToIndex>::value) &&
  (index_size<FromIndex>::value == 1),
  ToShape
>::type
  index_cast(const FromIndex& from_idx, const FromShape& from_shape, const ToShape& to_shape);


// recursive case for casting two indices of equal size
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value == index_size<ToIndex>::value) &&
  (index_size<FromIndex>::value > 1),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape& to_shape);


// upcast (recursive)
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value < index_size<ToIndex>::value),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape&   to_shape);


// downcast (recursive)
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value > index_size<ToIndex>::value),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape& to_shape);


// definitions of index_cast follow


// terminal case for casting indices of size 1
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value == index_size<ToIndex>::value) &&
  (index_size<FromIndex>::value == 1),
  ToShape
>::type
  index_cast(const FromIndex& from_idx, const FromShape&, const ToShape&)
{
  // from_idx might not be a tuple, but instead a scalar type
  // to ensure we can get the 0th value from from_idx in a uniform way, wrap it first
  return static_cast<ToIndex>(agency::get<0>(wrap_scalar(from_idx)));
}


struct index_cast_functor
{
  template<class ToIndex, class FromIndex, class FromShape, class ToShape>
  __AGENCY_ANNOTATION
  auto operator()(const ToIndex&, const FromIndex& from_idx, const FromShape& from_shape, const ToShape& to_shape)
    -> decltype(
          index_cast<ToIndex>(from_idx, from_shape, to_shape)
       )
  {
    return index_cast<ToIndex>(from_idx, from_shape, to_shape);
  }
};



// when both index types are the same size, index_cast recursively maps itself across the tuples
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value == index_size<ToIndex>::value) &&
  (index_size<FromIndex>::value > 1),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape& to_shape)
{
  return __tu::tuple_map_with_make(index_cast_functor{}, index_cast_detail::make<ToIndex>{}, ToIndex{}, from_idx, from_shape, to_shape);
}


// when FromIndex has fewer elements than ToIndex, we lift it and then cast
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value < index_size<ToIndex>::value),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape&   to_shape)
{
  auto lifted_idx_and_shape = lift_index(from_idx, from_shape, to_shape);
  return index_cast<ToIndex>(agency::get<0>(lifted_idx_and_shape), agency::get<1>(lifted_idx_and_shape), to_shape);
}


// when FromIndex has more elements than ToIndex, we project it and then cast
template<class ToIndex, class FromIndex, class FromShape, class ToShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (index_size<FromIndex>::value > index_size<ToIndex>::value),
  ToIndex
>::type
  index_cast(const FromIndex& from_idx,
             const FromShape& from_shape,
             const ToShape& to_shape)
{
  return index_cast<ToIndex>(detail::project_index(from_idx, from_shape), detail::project_shape(from_shape), to_shape);
}


// XXX this code is too confusing -- the stuff involved in lifting & projecting from/to arbitrary dimensions seems to be the problem
//     it seems like a simpler algorithm for index_cast would go something like the following:
//
//     1. find the lexicographic rank of from_idx in from_shape
//     2. return the index with corresponding lexicographic rank in to_shape

                     
} // end detail
} // end agency

