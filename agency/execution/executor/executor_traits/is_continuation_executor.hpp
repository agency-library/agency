#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/member_future_or.hpp>
#include <agency/future.hpp>
#include <future>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class Executor, class Function, class Future>
struct has_then_execute_impl
{
  using result_type = result_of_continuation_t<Function,Future>;
  using expected_future_type = member_future_or_t<Executor,result_type,std::future>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Future&>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_future_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class Function, class Future>
using has_then_execute = typename has_then_execute_impl<Executor, Function, Future>::type;


template<class T>
struct is_continuation_executor_impl
{
  // types related to functions passed to .then_execute()
  using result_type = int;
  using predecessor_type = int;
  using predecessor_future_type = member_future_or_t<T,predecessor_type,std::future>;

  // the functions we'll pass to .then_execute() to test

  // XXX WAR nvcc 8.0 bug
  //using test_function = std::function<result_type(predecessor_type&)>;

  struct test_function
  {
    result_type operator()(predecessor_type&);
  };

  using type = has_then_execute<
    T,
    test_function,
    predecessor_future_type
  >;
};


} // end detail


template<class T>
using is_continuation_executor = typename detail::is_continuation_executor_impl<T>::type;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool ContinuationExecutor()
{
  return is_continuation_executor<T>();
}


} // end detail
} // end agency

