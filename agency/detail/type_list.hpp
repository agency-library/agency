#pragma once

#include <type_traits>
#include <agency/detail/integer_sequence.hpp>

namespace agency
{
namespace detail
{


template<class... Types> struct type_list {};


template<class TypeList> struct type_list_size;


template<class... Types>
struct type_list_size<type_list<Types...>> : std::integral_constant<size_t, sizeof...(Types)> {};


template<size_t i, class TypeList>
struct type_list_element_impl;


template<class T0, class... Types>
struct type_list_element_impl<0,type_list<T0,Types...>>
{
  using type = T0;
};


template<size_t i, class T0, class... Types>
struct type_list_element_impl<i,type_list<T0,Types...>>
{
  using type = typename type_list_element_impl<i-1,type_list<Types...>>::type;
};


template<size_t i, class TypeList>
using type_list_element = typename type_list_element_impl<i,TypeList>::type;


// concatenate two type_lists
template<class TypeList1, class TypeList2> struct type_list_cat_impl;


template<class... Types1, class... Types2>
struct type_list_cat_impl<type_list<Types1...>, type_list<Types2...>>
{
  using type = type_list<Types1..., Types2...>;
};


template<class TypeList1, class TypeList2>
using type_list_cat = typename type_list_cat_impl<TypeList1,TypeList2>::type;


template<template<class> class MetaFunction, class TypeList>
struct type_list_filter_impl;

// an empty type_list filters to the empty type_list
template<template<class> class MetaFunction>
struct type_list_filter_impl<MetaFunction, type_list<>>
{
  using type = type_list<>;
};


template<template<class> class MetaFunction, class Type0, class... Types>
struct type_list_filter_impl<MetaFunction, type_list<Type0, Types...>>
{
  // recurse and filter the rest of the types
  using rest = typename type_list_filter_impl<MetaFunction, type_list<Types...>>::type;

  // concatenate Type0 with rest if Type0 passes the filter
  // else, just return rest
  using type = typename std::conditional<
    MetaFunction<Type0>::value,
    type_list_cat<
      type_list<Type0>,
      rest
    >,
    rest
  >::type;
};


template<template<class> class MetaFunction, class TypeList>
using type_list_filter = typename type_list_filter_impl<MetaFunction,TypeList>::type;


template<class TypeList, class IndexSequence>
struct type_list_take_impl_impl;


template<class TypeList, size_t... I>
struct type_list_take_impl_impl<TypeList, index_sequence<I...>>
{
  using type = type_list<
    type_list_element<I,TypeList>...
  >;
};


template<size_t n, class TypeList>
struct type_list_take_impl;


template<size_t n, class... Types>
struct type_list_take_impl<n,type_list<Types...>>
{
  using type = typename type_list_take_impl_impl<
    type_list<Types...>,
    make_index_sequence<n>
  >::type;
};


template<size_t n, class TypeList>
using type_list_take = typename type_list_take_impl<n,TypeList>::type;


namespace type_list_detail
{


template<int a, int b>
struct max : std::integral_constant<
  int,
  (a < b ? b : a)
>
{};


} // end type_list_detail


template<size_t n, class TypeList>
using type_list_drop = type_list_take<
  type_list_detail::max<
    0,
    type_list_size<TypeList>::value - n
  >::value,
  TypeList
>;


template<class TypeList>
using type_list_drop_last = type_list_drop<1,TypeList>;


template<class T, class TypeList>
struct is_constructible_from_type_list;


template<class T, class... Types>
struct is_constructible_from_type_list<T,type_list<Types...>>
  : std::is_constructible<T,Types...>
{};


template<class T0, class TypeList>
struct type_list_prepend;

template<class T0, class... Types>
struct type_list_prepend<T0, type_list<Types...>>
{
  using type = type_list<T0, Types...>;
};


} // end detail
} // end agency

