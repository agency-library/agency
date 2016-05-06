#pragma once

#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class T, class Executor, class... Args>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<T>
  make_ready_future(std::true_type, Executor& ex, Args&&... args)
{
  return ex.template make_ready_future<T>(std::forward<Args>(args)...);
} // end make_ready_future()


template<class T, class Executor, class... Args>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<T>
  make_ready_future(std::false_type, Executor&, Args&&... args)
{
  using future_type = typename executor_traits<Executor>::template future<T>;
  return future_traits<future_type>::template make_ready<T>(std::forward<Args>(args)...);
} // end make_ready_future()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class T, class... Args>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<T>
  executor_traits<Executor>
    ::make_ready_future(typename executor_traits<Executor>::executor_type& ex, Args&&... args)
{
  using check_for_member_function = agency::detail::executor_traits_detail::has_make_ready_future<
    Executor,
    T,
    Args&&...
  >;

  return detail::executor_traits_detail::make_ready_future<T>(check_for_member_function(), ex, std::forward<Args>(args)...);
} // end executor_traits::make_ready_future()


} // end agency

