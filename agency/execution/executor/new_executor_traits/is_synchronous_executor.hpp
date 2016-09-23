#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class Executor, class Function>
struct has_sync_execute_impl
{
  using expected_result_type = result_of_t<Function()>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().sync_execute(
               std::declval<Function>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_result_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class Function>
using has_sync_execute = typename has_sync_execute_impl<Executor, Function>::type;


template<class T>
struct is_synchronous_executor_impl
{
  // types related to functions passed to .sync_execute()
  using result_type = int;

  // the function we'll pass to .sync_execute() to test

  // XXX WAR nvcc 8.0 bug
  //using test_function = std::function<result_type()>;

  struct test_function
  {
    result_type operator()();
  };

  using type = has_sync_execute<
    T,
    test_function
  >;
};


} // end detail


template<class T>
using is_synchronous_executor = typename detail::is_synchronous_executor_impl<T>::type;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool SynchronousExecutor()
{
  return is_synchronous_executor<T>();
}


} // end detail
} // end agency

