#pragma once

#include <agency/future.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class U>
struct has_future_impl
{
  template<class> static std::false_type test(...);
  template<class X> static std::true_type test(typename X::template future<U>* = 0);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_future = typename has_future_impl<T,U>::type;


template<class Executor, class T, bool = has_future<Executor,T>::value>
struct executor_future
{
  using type = typename Executor::template future<T>;
};

template<class Executor, class T>
struct executor_future<Executor,T,false>
{
  using type = std::future<T>;
};


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


template<class Executor, class TupleOfFutures, class Function, size_t... Indices>
struct has_single_agent_when_all_execute_and_select_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<TupleOfFutures>(),
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class TupleOfFutures, class Function, size_t... Indices>
using has_single_agent_when_all_execute_and_select = typename has_single_agent_when_all_execute_and_select_impl<Executor, TupleOfFutures, Function, Indices...>::type;


template<class Executor, class TupleOfFutures, class Function>
struct has_single_agent_when_all_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all_execute(
               std::declval<TupleOfFutures>(),
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class TupleOfFutures, class Function>
using has_single_agent_when_all_execute = typename has_single_agent_when_all_execute_impl<Executor, TupleOfFutures, Function>::type;


template<class Executor, class... Futures>
struct has_when_all_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all(
               *std::declval<Futures*>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class... Futures>
using has_when_all = typename has_when_all_impl<Executor, Futures...>::type;


template<class Executor, class Future, class Function>
struct has_single_agent_then_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().then_execute(
               *std::declval<Future*>(),
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Future, class Function>
using has_single_agent_then_execute = typename has_single_agent_then_execute_impl<Executor,Future,Function>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
struct new_executor_traits
{
  public:
    using executor_type = Executor;

    template<class T>
    using future = typename detail::new_executor_traits_detail::executor_future<executor_type,T>::type;

    template<class T, class... Args>
    static future<T> make_ready_future(executor_type& ex, Args&&... args);

    template<size_t... Indices, class TupleOfFutures, class Function>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, TupleOfFutures&& futures, Function f);

    template<class Future, class Function>
    static future<
      typename detail::future_result_of<Function,Future>::type
    > then_execute(executor_type& ex, Future& fut, Function f);
}; // end new_executor_traits


} // end agency

#include <agency/detail/executor_traits/make_ready_future.hpp>
#include <agency/detail/executor_traits/when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/then_execute.hpp>

