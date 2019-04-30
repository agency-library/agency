#pragma once

#include <type_traits>
#include <tuple>
#include <agency/tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/coordinate.hpp>
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


template<class IndexTuple, class Size>
__AGENCY_ANNOTATION
auto project_index_helper(const IndexTuple& idx, Size size_of_lowest_dimension)
  -> typename std::decay<
       decltype(
         unwrap_single_element_tuple(detail::tuple_drop_last(idx))
       )
     >::type
{
  // drop the lowest dimension of the index to initialize the result
  auto result_idx = detail::tuple_drop_last(idx);

  // multiply the new lowest dimension of the resulting index by the size of the lowest dimension
  detail::tuple_last(result_idx) *= size_of_lowest_dimension;

  // and add in the old lowest dimension
  detail::tuple_last(result_idx) += detail::tuple_last(idx);

  // to understand why the above works, consider the order of iterations of a nested for loop
  // for(int i = 0; i < m; ++i)
  // {
  //   for(int j = 0; j < n; ++j)
  //   {
  //     // to project (i, j) into 1D:
  //     // multiply the new lowest dimension (i) by the size of the lowest dimension (n)
  //     // and add in the old lowest dimension (j)
  //     int rank = i * n + j;
  //
  //     // m isn't used at all
  //     // only the size of the lowest dimension is used
  //   }
  // }

  return unwrap_single_element_tuple(result_idx);
}


template<class IndexTuple, class ShapeTuple>
__AGENCY_ANNOTATION
auto project_index(const IndexTuple& idx, const ShapeTuple& shape)
  -> decltype(
       project_index_helper(idx, detail::tuple_last(detail::tuple_drop_last(shape)))
     )
{
  // the size of the lowest dimension is simply the last element of shape
  auto size_of_lowest_dimension = detail::tuple_last(shape);

  return detail::project_index_helper(idx, size_of_lowest_dimension);
}


namespace index_cast_detail
{


// to lift a point-like type, add a dimension
template<class Index>
struct lift_t_impl
{
  using type = rebind_point_size_t<Index, point_size<Index>::value + 1>;
};


// to lift a heterogeneous Tuple-like type, repeat the type of the last element
template<template<class...> class tuple, class T, class... Types>
struct lift_t_impl<tuple<T,Types...>>
{
  using last_type = typename std::tuple_element<sizeof...(Types)-1, std::tuple<Types...>>::type;
  using type = tuple<T,Types...,last_type>;
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
  // to lift idx into to_shape,
  // take the last element of idx and divide by the corresponding element of to_shape
  // replace the last element of idx with the remainder and append the quotient
  const auto i = index_size<Index>::value - 1;

  auto idx_tuple = wrap_scalar(idx);

  auto intermediate_result = idx_tuple;
  detail::tuple_last(intermediate_result) %= agency::get<i>(to_shape);

  auto index_maker = index_cast_detail::make<lift_t<Index>>{};

  auto lifted_index = __tu::tuple_append_invoke(intermediate_result, detail::tuple_last(idx_tuple) / agency::get<i>(to_shape), index_maker);

  // to lift from_shape, simply append the element of to_shape we just divided by
  auto shape_maker = index_cast_detail::make<lift_t<FromShape>>{};
  auto lifted_shape = __tu::tuple_append_invoke(wrap_scalar(from_shape), agency::get<i>(to_shape), shape_maker);

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

                     
} // end detail
} // end agency

