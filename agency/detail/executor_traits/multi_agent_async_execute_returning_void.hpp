#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/invoke_and_return_empty.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Function>
typename executor_traits<Executor>::template future<void>
  multi_agent_async_execute_returning_void(std::true_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  return ex.async_execute(f, shape);
} // end multi_agent_async_execute_returning_void()


template<class Executor, class Function>
typename executor_traits<Executor>::template future<void>
  multi_agent_async_execute_returning_void(std::false_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = executor_traits<Executor>::async_execute(ex, invoke_and_return_empty<Function>{f}, shape);

  // cast the discarding_container to void
  return executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_async_execute_returning_void()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename executor_traits<Executor>::template future<void>
  executor_traits<Executor>
    ::async_execute(typename executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::executor_traits_detail::has_multi_agent_async_execute_returning_void<
    Executor,
    Function
  >;

  return detail::executor_traits_detail::multi_agent_async_execute_returning_void(check_for_member_function(), ex, f, shape);
} // end executor_traits::async_execute()


} // end agency

