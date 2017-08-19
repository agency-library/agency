#pragma once

#include <agency/detail/config.hpp>

// XXX this #include should be eliminated from this file
#define TUPLE_UTILITY_ANNOTATION __AGENCY_ANNOTATION
#define TUPLE_UTILITY_NAMESPACE __tu
#include <agency/tuple/detail/tuple_utility_impl.hpp>
#undef TUPLE_UTILITY_ANNOTATION
#undef TUPLE_UTILITY_NAMESPACE


#include <agency/detail/requires.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/host_device_cast.hpp>
#include <agency/detail/has_member.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


__DEFINE_HAS_MEMBER(has_value, value);


template<class T>
struct is_tuple : has_value<std::tuple_size<T>> {};


// fancy version of std::get which uses tuple_traits and can get() from things which aren't in std::
template<size_t i, class Tuple,
         class = typename std::enable_if<
           is_tuple<typename std::decay<Tuple>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto get(Tuple&& t)
  -> decltype(
       __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t))
     )
{
  return __tu::tuple_traits<typename std::decay<Tuple>::type>::template get<i>(std::forward<Tuple>(t));
}


// get_if returns the ith element of an object when that object is a Tuple-like type
// otherwise, it returns its second parameter
template<size_t i, class Tuple, class T,
         __AGENCY_REQUIRES(
           is_tuple<typename std::decay<Tuple>::type>::value
         )>
__AGENCY_ANNOTATION
auto get_if(Tuple&& t, T&&)
  -> decltype(get<i>(std::forward<Tuple>(t)))
{
  return detail::get<i>(std::forward<Tuple>(t));
}


template<size_t, class NotATuple, class T,
         __AGENCY_REQUIRES(
           !is_tuple<typename std::decay<NotATuple>::type>::value
         )>
__AGENCY_ANNOTATION
T&& get_if(NotATuple&&, T&& otherwise_if_not_tuple)
{
  return std::forward<T>(otherwise_if_not_tuple);
}


// names the ith type of a parameter pack
template<size_t i, class... Types>
struct pack_element
  : std::tuple_element<i,std::tuple<Types...>>
{
};


template<size_t i, class... Types>
using pack_element_t = typename pack_element<i,Types...>::type;


template<class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_head_if(T&& t) ->
  decltype(detail::get<0>(std::forward<T>(t)))
{
  return detail::get<0>(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_head_if(T&& t)
{
  return std::forward<T>(t);
}


template<class T,
         class = typename std::enable_if<
           is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto tuple_last_if(T&& t) ->
  decltype(__tu::tuple_last(std::forward<T>(t)))
{
  return __tu::tuple_last(std::forward<T>(t));
}


template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
__AGENCY_ANNOTATION
T&& tuple_last_if(T&& t)
{
  return std::forward<T>(t);
}


template<class Function, class Tuple>
__AGENCY_ANNOTATION
auto tuple_apply(Function&& f, Tuple&& t)
  -> decltype(
       __tu::tuple_apply(agency::detail::host_device_cast(std::forward<Function>(f)), std::forward<Tuple>(t))
     )
{
  return __tu::tuple_apply(agency::detail::host_device_cast(std::forward<Function>(f)), std::forward<Tuple>(t));
}


template<class Function>
struct tuple_all_of_functor
{
  Function f;

  template<class Arg>
  __AGENCY_ANNOTATION
  bool operator()(bool prefix, Arg&& arg) const
  {
    return prefix && f(std::forward<Arg>(arg));
  }
};


template<class Tuple, class Function>
__AGENCY_ANNOTATION
bool tuple_all_of(Tuple&& t, Function f)
{
  return __tu::tuple_reduce(std::forward<Tuple>(t), true, tuple_all_of_functor<Function>{f});
}


template<class Tuple>
using tuple_indices = make_index_sequence<std::tuple_size<Tuple>::value>;


template<class Tuple>
__AGENCY_ANNOTATION
detail::make_index_sequence<
  std::tuple_size<
    typename std::decay<Tuple>::type
  >::value
> 
  make_tuple_indices(Tuple&&)
{
  return detail::make_index_sequence<
    std::tuple_size<
      typename std::decay<Tuple>::type
    >::value
  >();
}


template<class IndexSequence, class Tuple>
struct tuple_elements_impl;

template<size_t... Indices, class Tuple>
struct tuple_elements_impl<index_sequence<Indices...>,Tuple>
{
  using type = type_list<
    typename std::tuple_element<Indices,Tuple>::type...
  >;
};


template<class Tuple>
using tuple_elements = typename tuple_elements_impl<tuple_indices<Tuple>,Tuple>::type;


template<class Tuple,
         class = typename std::enable_if<
           (std::tuple_size<
             typename std::decay<Tuple>::type
           >::value > 1)
         >::type
        >
__AGENCY_ANNOTATION
Tuple&& unwrap_single_element_tuple(Tuple&& t)
{
  return std::forward<Tuple>(t);
}


template<class Tuple,
         class = typename std::enable_if<
           (std::tuple_size<
              typename std::decay<Tuple>::type
           >::value == 1)
         >::type
        >
__AGENCY_ANNOTATION
auto unwrap_single_element_tuple(Tuple&& t)
  -> decltype(
       detail::get<0>(std::forward<Tuple>(t))
     )
{
  return detail::get<0>(std::forward<Tuple>(t));
}


// if the argument is a tuple, it unwraps it if it is a single-element tuple,
// otherwise, it returns the tuple
// if the argument is not a tuple, it returns the argument
template<class Tuple,
         class = typename std::enable_if<
           is_tuple<typename std::decay<Tuple>::type>::value
         >::type>
__AGENCY_ANNOTATION
auto unwrap_single_element_tuple_if(Tuple&& t)
  -> decltype(
       detail::unwrap_single_element_tuple(std::forward<Tuple>(t))
     )
{
  return detail::unwrap_single_element_tuple(std::forward<Tuple>(t));
}

template<class T,
         class = typename std::enable_if<
           !is_tuple<typename std::decay<T>::type>::value
         >::type>
T&& unwrap_single_element_tuple_if(T&& arg)
{
  return std::forward<T>(arg);
}


template<class Indices, class Tuple>
struct tuple_type_list_impl;

template<size_t... Indices, class Tuple>
struct tuple_type_list_impl<index_sequence<Indices...>, Tuple>
{
  using type = type_list<
    typename std::tuple_element<Indices,Tuple>::type...
  >;
};


template<class T, class Enable = void>
struct tuple_type_list;


template<class Tuple>
struct tuple_type_list<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
{
  using type = typename tuple_type_list_impl<
    make_index_sequence<std::tuple_size<Tuple>::value>,
    Tuple
  >::type;
};


template<class>
struct is_empty_tuple;


template<class T>
struct is_empty_tuple_impl_impl;


template<class... Types>
struct is_empty_tuple_impl_impl<type_list<Types...>>
{
  using type = conjunction<
    disjunction<
      std::is_empty<Types>,
      is_empty_tuple<Types>
    >...
  >;
};


template<class T, class Enable = void>
struct is_empty_tuple_impl : std::false_type {};


template<class Tuple>
struct is_empty_tuple_impl<Tuple, typename std::enable_if<is_tuple<Tuple>::value>::type>
{
  using type = typename is_empty_tuple_impl_impl<
    typename tuple_type_list<Tuple>::type
  >::type;
};


template<class Tuple>
struct is_empty_tuple : is_empty_tuple_impl<Tuple>::type {};


// tuple_rebind takes a Tuple-like type and reinstantiates it with a different list of types
template<class Tuple, class... Types>
struct tuple_rebind;


// we can tuple_rebind a Tuple-like type simply by reinstantiating the template from which it came
template<template<class...> class TupleLike, class... OriginalTypes, class... Types>
struct tuple_rebind<TupleLike<OriginalTypes...>, Types...>
{
  using type = TupleLike<Types...>;
};



// we can tuple_rebind an Array-like type only when the list of Types are all the same type
template<template<class,size_t> class ArrayLike, class OriginalType, size_t n, class Type, class... Types>
struct tuple_rebind<ArrayLike<OriginalType,n>, Type, Types...>
  : std::conditional<
      conjunction<std::is_same<Type,Types>...>::value,  // if all of Types are the same as Type
      ArrayLike<Type, 1 + sizeof...(Types)>,            // then reinstantiate the Array-like template using Type
      std::enable_if<false>                             // otherwise, do not define a member named ::type
    >::type
{};


template<class Tuple, class... Types>
using tuple_rebind_t = typename tuple_rebind<Tuple,Types...>::type;


// a Tuple-like type is rebindable for a list of types if tuple_rebind<Tuple,Types...>::type is detected to exist
// XXX WAR nvbug 1965139
//template<class Tuple, class... Types>
//using is_tuple_rebindable = is_detected<tuple_rebind_t, Tuple, Types...>;
template<class Tuple, class... Types>
struct is_tuple_rebindable : is_detected<tuple_rebind_t, Tuple, Types...> {};


// some types aren't tuple_rebindable given a list of Types
// in such cases, we default to using the given TupleLike template as the result of the rebind
template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if = lazy_conditional<
  is_tuple_rebindable<T,Types...>::value, // if Tuple is rebindable...
  tuple_rebind<T,Types...>,               // then tuple_rebind it
  identity<TupleLike<Types...>>           // otherwise, default to TupleLike<Types...>
>;

template<class T, template<class...> class TupleLike, class... Types>
using tuple_rebind_if_t = typename tuple_rebind_if<T,TupleLike,Types...>::type;


} // end detail
} // end agency

