#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::true_type, Executor& ex, Function f)
{
  return ex.async_execute(f);
} // end single_agent_async_execute()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::false_type, Executor& ex, Function f)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::then_execute(ex, f, ready);
} // end single_agent_async_execute()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_async_execute<
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::single_agent_async_execute(check_for_member_function(), ex, f);
} // end new_executor_traits::async_execute()


} // end agency

