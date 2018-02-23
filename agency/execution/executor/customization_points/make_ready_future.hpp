#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class Executor, class T, class... Args>
struct has_make_ready_future_impl
{
  template<
    class Executor2,
    typename = decltype(
      std::declval<Executor2&>().template make_ready_future<T>(
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


// this overload handles the case of executors which have the member function .make_ready_future()
__agency_exec_check_disable__
template<class T, class Executor, class... Args>
__AGENCY_ANNOTATION
executor_future_t<Executor,T>
  make_ready_future_impl(std::true_type, const Executor& exec, Args&&... args)
{
  return exec.template make_ready_future<T>(std::forward<Args>(args)...);
} // end make_ready_future_impl()


// this overload handles the case of executors which do not have the member function .make_ready_future()
template<class T, class Executor, class... Args>
__AGENCY_ANNOTATION
executor_future_t<Executor,T>
  make_ready_future_impl(std::false_type, const Executor&, Args&&... args)
{
  using future_type = executor_future_t<Executor,T>;
  return future_traits<future_type>::template make_ready<T>(std::forward<Args>(args)...);
} // end make_ready_future_impl()


} // end detail


template<class T, class E, class... Args,
         __AGENCY_REQUIRES(is_executor<E>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,T> make_ready_future(const E& exec, Args&&... args)
{
  using check_for_member_function = detail::has_make_ready_future<
    E,
    T,
    Args&&...
  >;

  return detail::make_ready_future_impl<T>(check_for_member_function(), exec, std::forward<Args>(args)...);
} // end make_ready_future()


} // end agency

