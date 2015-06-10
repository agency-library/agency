#pragma once

#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


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


template<class T, class Executor, class... Args>
typename new_executor_traits<Executor>::template future<T>
  make_ready_future(std::true_type, Executor& ex, Args&&... args)
{
  return ex.template make_ready_future<T>(std::forward<Args>(args)...);
} // end make_ready_future()


template<class T, class Executor, class... Args>
typename new_executor_traits<Executor>::template future<T>
  make_ready_future(std::false_type, Executor&, Args&&... args)
{
  using future_type = typename new_executor_traits<Executor>::template future<T>;
  return future_traits<future_type>::template make_ready<T>(std::forward<Args>(args)...);
} // end make_ready_future()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class T, class... Args>
typename new_executor_traits<Executor>::template future<T>
  new_executor_traits<Executor>
    ::make_ready_future(typename new_executor_traits<Executor>::executor_type& ex, Args&&... args)
{
  using check_for_member_function = agency::detail::new_executor_traits_detail::has_make_ready_future<
    Executor,
    T,
    Args&&...
  >;

  return detail::new_executor_traits_detail::make_ready_future<T>(check_for_member_function(), ex, std::forward<Args>(args)...);
} // end new_executor_traits::make_ready_future()


} // end agency

