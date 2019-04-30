#pragma once

#include <type_traits>
#include <agency/coordinate.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>

namespace agency
{
namespace detail
{
namespace point_size_detail
{


template<class T, size_t N, class Enable = void> struct rebind_point_size_impl;

template<template<class,size_t> class Point, class T, size_t N, size_t M>
struct rebind_point_size_impl<Point<T,N>, M>
{
  using type = Point<T,M>;
};

template<template<class,int> class Point, class T, int N, size_t M>
struct rebind_point_size_impl<Point<T,N>, M>
{
  using type = Point<T,M>;
};

template<template<size_t,class> class Point, size_t N, class T, size_t M>
struct rebind_point_size_impl<Point<N,T>, M>
{
  using type = Point<M,T>;
};

template<template<int,class> class Point, int N, class T, size_t M>
struct rebind_point_size_impl<Point<N,T>, M>
{
  using type = Point<M,T>;
};


template<class T, size_t N>
struct rebind_point_size_impl<T, N,
  typename std::enable_if<
    std::is_arithmetic<T>::value
  >::type
>
{
  using type = agency::point<T,N>;
};


template<class T, class Tuple>
struct append_tuple_element;

template<class T, template<class...> class tuple_like, class... Types>
struct append_tuple_element<T, tuple_like<Types...>>
{
  using type = tuple_like<Types..., T>;
};


template<template<class...> class tuple_like, class T, size_t N>
struct make_homogeneous_tuple
{
  using rest = typename make_homogeneous_tuple<tuple_like, T, N - 1>::type;

  using type = typename append_tuple_element<
    T,
    rest
  >::type;
};

template<template<class...> class tuple_like, class T>
struct make_homogeneous_tuple<tuple_like, T, 0>
{
  using type = tuple_like<>;
};


} // end point_size_detail


template<class T, size_t N>
struct rebind_point_size
{
  using type = typename point_size_detail::rebind_point_size_impl<T,N>::type;
};

// specialization for types which are instances of Tuple-like templates
// XXX should actually enable only if tuple_like<T,Types...> is Tuple-like
template<template<class...> class tuple_like, class T, class... Types, size_t N>
struct rebind_point_size<tuple_like<T, Types...>, N>
{
  using type = typename point_size_detail::make_homogeneous_tuple<tuple_like, T, N>::type;
};


template<class T, size_t N>
using rebind_point_size_t = typename rebind_point_size<T,N>::type;


template<class T, class Enable = void>
struct point_size : std::tuple_size<T> {};


template<class T>
struct point_size<
  T,
  typename std::enable_if<
    std::is_arithmetic<T>::value
  >::type
>
  : std::integral_constant<std::size_t, 1>
{};


} // end detail
} // end agency

