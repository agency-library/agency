#pragma once

#include <type_traits>
#include <agency/coordinate.hpp>
#include <agency/detail/tuple_utility.hpp>

namespace agency
{
namespace detail
{
namespace point_size_detail
{


template<class T, size_t N, class Enable = void> struct rebind_point_size_impl;

template<template<class,size_t> class point, class T, size_t N, size_t M>
struct rebind_point_size_impl<point<T,N>, M>
{
  using type = point<T,M>;
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


} // end point_size_detail


template<class T, size_t N>
struct rebind_point_size
{
  using type = typename point_size_detail::rebind_point_size_impl<T,N>::type;
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

