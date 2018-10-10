#pragma once

#include <type_traits>
#include <agency/detail/integer_sequence.hpp>

namespace agency
{
namespace detail
{


template<bool b, class T, class F>
using conditional_t = typename std::conditional<b,T,F>::type;


template<bool b, typename T, typename F>
struct lazy_conditional
{
  using type = typename T::type;
};


template<typename T, typename F>
struct lazy_conditional<false,T,F>
{
  using type = typename F::type;
};


template<bool condition, typename T, typename F>
using lazy_conditional_t = typename lazy_conditional<condition, T, F>::type;


template<typename T>
struct identity
{
  typedef T type;
};


template<class T>
using decay_t = typename std::decay<T>::type;


template<class... Conditions>
struct conjunction;

template<>
struct conjunction<> : std::true_type {};

template<class Condition, class... Conditions>
struct conjunction<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value && conjunction<Conditions...>::value
    >
{};


template<class... Conditions>
struct disjunction;

template<>
struct disjunction<> : std::false_type {};

template<class Condition, class... Conditions>
struct disjunction<Condition, Conditions...>
  : std::integral_constant<
      bool,
      Condition::value || disjunction<Conditions...>::value
    >
{};


template<class T>
struct lazy_add_lvalue_reference
{
  using type = typename std::add_lvalue_reference<typename T::type>::type;
};


template<class T, template<class...> class Template>
struct is_instance_of : std::false_type {};


template<class... Types, template<class...> class Template>
struct is_instance_of<Template<Types...>,Template> : std::true_type {};


template<class Reference, class T>
struct propagate_reference;

template<class U, class T>
struct propagate_reference<U&, T>
{
  using type = T&;
};

template<class U, class T>
struct propagate_reference<const U&, T>
{
  using type = const T&;
};

template<class U, class T>
struct propagate_reference<U&&, T>
{
  using type = T&&;
};

template<class Reference, class T>
using propagate_reference_t = typename propagate_reference<Reference,T>::type;


template<class T>
struct is_not_void : std::integral_constant<bool, !std::is_void<T>::value> {};


template<class T>
struct decay_if_not_void : std::decay<T> {};

template<>
struct decay_if_not_void<void>
{
  using type = void;
};

template<class T>
using decay_if_not_void_t = typename decay_if_not_void<T>::type;


template<class...> 
using void_t = void; 
 
struct nonesuch 
{ 
  nonesuch() = delete; 
  ~nonesuch() = delete; 
  nonesuch(const nonesuch&) = delete; 
  void operator=(const nonesuch&) = delete; 
}; 
 
 
template<class Default, class AlwaysVoid,
         template<class...> class Op, class... Args> 
struct detector 
{ 
  using value_t = std::false_type; 
  using type = Default; 
}; 
 
 
template<class Default, template<class...> class Op, class... Args> 
struct detector<Default, void_t<Op<Args...>>, Op, Args...> 
{ 
  using value_t = std::true_type; 
  using type = Op<Args...>; 
}; 
 
 
template<template<class...> class Op, class... Args> 
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t; 
 
template<template<class...> class Op, class... Args> 
using detected_t = typename detector<nonesuch, void, Op, Args...>::type; 

template<class Expected, template<class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op,Args...>>;
 
template<class Default, template<class...> class Op, class... Args> 
using detected_or = detector<Default, void, Op, Args...>; 
 
template<class Default, template<class...> class Op, class... Args> 
using detected_or_t = typename detected_or<Default,Op,Args...>::type; 


template<class T>
struct is_cuda_extended_device_lambda
  : 
#if __CUDACC_EXTENDED_LAMBDA__
    std::integral_constant<bool, __nv_is_extended_device_lambda_closure_type(T)>
#else
    std::false_type
#endif
{};


template<class T, class Enable = void>
struct result_of_impl : std::result_of<T> {};

template<class Function, class... Args>
struct result_of_impl<
  Function(Args...),
  typename std::enable_if<
    is_cuda_extended_device_lambda<Function>::value
  >::type
>
{
  // XXX we should actually test that Function is callable with Args...
  //     and then only include this using declaration if it is callable
  using type = void;
};


template<class T>
struct result_of : result_of_impl<T> {};


template<class T>
using result_of_t = typename result_of<T>::type;


template<class T1, class T2>
using is_not_same = std::integral_constant<bool, !std::is_same<T1,T2>::value>;


template<class T, class... Args>
struct is_constructible_or_void
  : std::integral_constant<
      bool,
      std::is_constructible<T,Args...>::value ||
      (std::is_void<T>::value && (sizeof...(Args) == 0))
    >
{};


} // end detail
} // end agency

