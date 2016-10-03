#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/member_future_or.hpp>
#include <future>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class Executor, class Function>
struct has_async_execute_impl
{
  using result_type = result_of_t<Function()>;
  using expected_future_type = member_future_or_t<Executor,result_type,std::future>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>()
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


template<class Executor, class Function>
using has_async_execute = typename has_async_execute_impl<Executor, Function>::type;


template<class T>
struct is_asynchronous_executor_impl
{
  // types related to functions passed to .async_execute()
  using result_type = int;

  // the function we'll pass to .async_execute() to test

  // XXX WAR nvcc 8.0 bug
  //using test_function = std::function<result_type()>;

  struct test_function
  {
    result_type operator()();
  };

  using type = has_async_execute<
    T,
    test_function
  >;
};


} // end detail


template<class T>
using is_asynchronous_executor = typename detail::is_asynchronous_executor_impl<T>::type;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool AsynchronousExecutor()
{
  return is_asynchronous_executor<T>();
}


} // end detail
} // end agency

