#pragma once

#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/factory.hpp>


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
struct has_any_multi_agent_then_execute_impl;


template<class Executor, class T, class... Factories>
struct has_any_multi_agent_then_execute_impl<Executor, T, type_list<Factories...>>
{
  template<class U>
  using future = typename nested_future_with_default<
    Executor
  >::template type_template<U>;

  static constexpr bool has_then_execute_with_shared_inits_returning_user_specified_container = new_executor_traits_detail::has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<
    new_executor_traits_detail::discarding_container,
    Executor,
    new_executor_traits_detail::test_function_returning_int,
    future<void>,
    Factories...
  >::value;

  static constexpr bool has_then_execute_with_shared_inits_returning_void = new_executor_traits_detail::has_multi_agent_then_execute_with_shared_inits_returning_void<
    Executor,
    new_executor_traits_detail::test_function_returning_void,
    future<void>,
    Factories...
  >::value;

  using type = std::integral_constant<
    bool,
    has_then_execute_with_shared_inits_returning_user_specified_container ||
    has_then_execute_with_shared_inits_returning_void
  >;
};



template<class T>
struct has_any_multi_agent_then_execute
  : has_any_multi_agent_then_execute_impl<
      T,
      int,
      repeat_type<
        unit_factory, execution_depth<typename T::execution_category>::value
      >
    >::type
{};


} // end detail


// XXX is_executor should be much more permissive and just check for any function supported by executor_traits
template<class T>
struct is_executor
  : std::integral_constant<
      bool,
      detail::new_executor_traits_detail::has_execution_category<T>::value &&
      detail::has_any_multi_agent_then_execute<T>::value
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

