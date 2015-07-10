#pragma once

#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/bind.hpp>
#include <agency/execution_categories.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/new_executor_traits.hpp>


namespace agency
{
namespace detail
{


template<class T>
struct nested_future_template
{
  template<class U>
  using type_template = typename T::template future<U>;
};


template<bool condition, class Then, class Else>
struct lazy_conditional_template
{
  template<class T>
  using type_template = typename Then::template type_template<T>;
};

template<class Then, class Else>
struct lazy_conditional_template<false, Then, Else>
{
  template<class T>
  using type_template = typename Else::template type_template<T>;
};


template<template<class> class T>
struct identity_template
{
  template<class U>
  using type_template = T<U>;
};


template<class T, template<class> class Default = std::future>
struct nested_future_with_default
  : agency::detail::lazy_conditional_template<
      agency::detail::new_executor_traits_detail::has_future<T,void>::value,
      nested_future_template<T>,
      identity_template<Default>
    >
{};


template<class Executor, class T, class TypeList>
struct has_multi_agent_then_execute_impl;


template<class Executor, class T, class... Types>
struct has_multi_agent_then_execute_impl<Executor, T, type_list<Types...>>
{
  using index_type = typename new_executor_traits_detail::nested_index_type_with_default<
    Executor
  >::type;

  using shape_type = typename new_executor_traits_detail::nested_shape_type_with_default<
    Executor
  >::type;

  template<class U>
  using future = typename nested_future_with_default<
    Executor
  >::template type_template<U>;

  struct dummy_functor
  {
    void operator()(const index_type&, T&, Types&...) {}

    // overload for void past parameter
    void operator()(const index_type&, Types&...) {}
  };

  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<dummy_functor>(),
               std::declval<shape_type>(),
               *std::declval<future<T>*>(),
               *std::declval<Types*>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class T, bool = new_executor_traits_detail::has_execution_category<T>::value>
struct has_multi_agent_then_execute : std::false_type {};

template<class T>
struct has_multi_agent_then_execute<T,true> 
  : has_multi_agent_then_execute_impl<
      T,
      int,
      repeat_type<
        int, execution_depth<typename T::execution_category>::value
      >
    >::type
{};


template<class Executor, class T, class... Args>
struct has_make_ready_future_impl
{
  template<
    class Executor2,
    typename = decltype(
      std::declval<Executor2*>()->template make_ready_future<T>(
        std::declval<Args>()...
      )
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class T, class... Args>
using has_make_ready_future = typename has_make_ready_future_impl<Executor,T,Args...>::type;


} // end detail


template<class T>
struct is_executor
  : std::integral_constant<
      bool,
      detail::new_executor_traits_detail::has_execution_category<T>::value &&
      detail::has_multi_agent_then_execute<T>::value
    >
{};


template<class Executor>
using executor_traits = new_executor_traits<Executor>;


namespace detail
{


template<class Executor, class Enable = void>
struct executor_index {};

template<class Executor>
struct executor_index<Executor, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::index_type;
};

template<class Executor>
using executor_index_t = typename executor_index<Executor>::type;


template<class Executor, class Enable = void>
struct executor_shape {};

template<class Executor>
struct executor_shape<Executor, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::shape_type;
};

template<class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


template<class Executor, class T, class Enable = void>
struct executor_future {};

template<class Executor, class T>
struct executor_future<Executor, T, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::template future<T>;
};

template<class Executor, class T>
using executor_future_t = typename executor_future<Executor,T>::type;


} // end detail


} // end agency

