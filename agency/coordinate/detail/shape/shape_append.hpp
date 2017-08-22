#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/tuple.hpp>
#include <agency/coordinate/detail/shape/shape_size.hpp>
#include <agency/coordinate/detail/shape/shape_element.hpp>

namespace agency
{
namespace detail
{


template<class Indices, class Shape, class T>
struct shape_append_t_impl_general_case;


// to append a type T to a Shape, create a tuple of the Shape's elements, and then add T at the end of this tuple
template<size_t... Indices, class Shape, class T>
struct shape_append_t_impl_general_case<index_sequence<Indices...>, Shape, T>
{
  using type = agency::tuple<shape_element_t<Indices,Shape>..., T>;
};


template<class Shape, class T>
struct shape_append_t_impl
{
  using type = typename shape_append_t_impl_general_case<
    make_index_sequence<shape_size<Shape>::value>,
    Shape,
    T
  >::type;
};


// When the originating Shape to extend is an instantiation of some template with a familiar signature, these specializations below preserve that template in the instantiation of the result

// to append a type T to an Array of Ts, just extend the Array's length by 1
template<template<class, size_t> class ArrayLike, size_t n, class T>
struct shape_append_t_impl<ArrayLike<T,n>, T>
{
  using type = ArrayLike<T, n+1>;
};


// to append a type T to a Tuple, just introduce a T at the end of the Tuple
template<template<class...> class TupleLike, class... TupleElements, class T>
struct shape_append_t_impl<TupleLike<TupleElements...>, T>
{
  using type = TupleLike<TupleElements..., T>;
};


// to append a type T to a std::pair, create a std::tuple
// we have to introduce this specialization otherwise it is caught by the specialization above
template<class T1, class T2, class T3>
struct shape_append_t_impl<std::pair<T1,T2>, T3>
{
  using type = std::tuple<T1,T2,T3>;
};


template<class Shape, class T>
using shape_append_t = typename shape_append_t_impl<Shape,T>::type;


// the reason that there is no type trait named shape_append is to reserve this name for the hypothetical function:
//
// template<class Shape, class T>
// shape_append_t<Shape,T> shape_append(const Shape& shape, const T& value);


} // end detail
} // end agency

