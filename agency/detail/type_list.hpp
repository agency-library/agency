#pragma once

#include <type_traits>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>

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
struct type_list_map_impl;

template<template<class> class MetaFunction, class... Types>
struct type_list_map_impl<MetaFunction,type_list<Types...>>
{
  using type = type_list<
    typename MetaFunction<Types>::type...
  >;
};


template<template<class> class MetaFunction, class... Types>
using type_list_map = typename type_list_map_impl<MetaFunction,Types...>::type;


template<class IndexSequence, class TypeList>
struct type_list_gather_impl;

template<size_t... Indices, class... Types>
struct type_list_gather_impl<index_sequence<Indices...>,type_list<Types...>>
{
  using type = type_list<
    type_list_element<Indices,type_list<Types...>>...
  >;
};

template<class IndexSequence, class TypeList>
using type_list_gather = typename type_list_gather_impl<IndexSequence,TypeList>::type;


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


template<class TypeList, class T>
struct type_list_append_impl;

template<class... Types, class T>
struct type_list_append_impl<type_list<Types...>, T>
{
  using type = type_list<Types..., T>;
};

template<class TypeList, class T>
using type_list_append = typename type_list_append_impl<TypeList, T>::type;


template<class Function, class TypeList>
struct type_list_is_callable_impl;

template<class Function, class... Types>
struct type_list_is_callable_impl<Function, type_list<Types...>>
{
  template<class Function1,
           class = decltype(
             std::declval<Function1>()(
               std::declval<Types>()...
             )
           )
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Function>(0));
};


template<class Function, class TypeList>
using type_list_is_callable = typename type_list_is_callable_impl<Function,TypeList>::type;


template<class Function, class TypeList>
struct type_list_result_of;

template<class Function, class... Types>
struct type_list_result_of<Function, type_list<Types...>>
{
  using type = result_of_t<Function(Types...)>;
};

template<class Function, class TypeList>
using type_list_result_of_t = typename type_list_result_of<Function,TypeList>::type;


template<size_t n, class T>
struct type_list_repeat_impl
{
  using rest = typename type_list_repeat_impl<n-1,T>::type;
  using type = type_list_append<rest, T>;
};

template<class T>
struct type_list_repeat_impl<0,T>
{
  using type = type_list<>;
};

template<size_t n, class T>
using type_list_repeat = typename type_list_repeat_impl<n,T>::type;


template<class Integer, template<class> class MetaFunction, class TypeList>
struct type_list_integer_map_impl;

template<class Integer, template<class> class MetaFunction, class... Types>
struct type_list_integer_map_impl<Integer,MetaFunction,type_list<Types...>>
{
  using type = integer_sequence<
    Integer,
    MetaFunction<Types>::value...
  >;
};


template<class Integer, template<class> class MetaFunction, class... Types>
using type_list_integer_map = typename type_list_integer_map_impl<Integer, MetaFunction,Types...>::type;

template<template<class> class MetaFunction, class... Types>
using type_list_index_map = type_list_integer_map<size_t, MetaFunction, Types...>;


template<class TypeList>
struct type_list_indices_impl;

template<class... Types>
struct type_list_indices_impl<type_list<Types...>>
{
  using type = make_index_sequence<sizeof...(Types)>;
};

template<class TypeList>
using type_list_indices = typename type_list_indices_impl<TypeList>::type;


template<class IndexSequence, class TypeList>
struct type_list_reverse_impl;

template<size_t... Indices, class TypeList>
struct type_list_reverse_impl<index_sequence<Indices...>, TypeList>
{
  using type = type_list<
    type_list_element<type_list_size<TypeList>::value - Indices - 1, TypeList>...
  >;
};

template<class TypeList>
using type_list_reverse = typename type_list_reverse_impl<type_list_indices<TypeList>,TypeList>::type;


template<template<class... Params> class Template, class TypeList>
struct type_list_instantiate_impl;

// XXX we should really enable_if sizeof...(Params) == sizeof...(Types)
template<template<class... Params> class Template, class... Types>
struct type_list_instantiate_impl<Template,type_list<Types...>>
{
  using type = Template<Types...>;
};

// type_list_instantiate takes a class template and a list of types as a type_list
// and returns that class template instantiated with the types in the type_list
// note that this is the inverse of type_list_of_template_parameters
template<template<class... Params> class Template, class TypeList>
using type_list_instantiate = typename type_list_instantiate_impl<Template,TypeList>::type;


template<class TypeList>
struct type_list_tail_impl;

template<class Type1, class... Types>
struct type_list_tail_impl<type_list<Type1,Types...>>
{
  using type = type_list<Types...>;
};

template<class TypeList>
using type_list_tail = typename type_list_tail_impl<TypeList>::type;


template<class T>
struct type_list_of_template_parameters_impl;

template<template<class... Params> class Template, class... Types>
struct type_list_of_template_parameters_impl<Template<Types...>>
{
  using type = type_list<Types...>;
};

// type_list_of_template_parameters takes an instance of a template
// and returns the list of types used to instantiate that template in a type_list
// note that this is the inverse of type_list_instantiate
template<class T>
using type_list_of_template_parameters = typename type_list_of_template_parameters_impl<T>::type;


} // end detail
} // end agency

