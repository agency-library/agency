#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Function>
typename executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::true_type, Executor& ex, Function&& f)
{
  return ex.async_execute(std::forward<Function>(f));
} // end single_agent_async_execute()


template<class Executor, class Function>
typename executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::false_type, Executor& ex, Function&& f)
{
  auto ready = executor_traits<Executor>::template make_ready_future<void>(ex);
  return executor_traits<Executor>::then_execute(ex, std::forward<Function>(f), ready);
} // end single_agent_async_execute()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function>
typename executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  executor_traits<Executor>
    ::async_execute(typename executor_traits<Executor>::executor_type& ex,
                    Function&& f)
{
  using check_for_member_function = detail::executor_traits_detail::has_single_agent_async_execute<
    Executor,
    Function
  >;

  return detail::executor_traits_detail::single_agent_async_execute(check_for_member_function(), ex, std::forward<Function>(f));
} // end executor_traits::async_execute()


} // end agency

