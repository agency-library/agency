#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/invoke_and_return_empty.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_async_execute_with_shared_inits_returning_void(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return ex.async_execute(f, shape, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_void()


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_async_execute_with_shared_inits_returning_void(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = new_executor_traits<Executor>::async_execute(ex, invoke_and_return_empty<Function>{f}, container_factory<discarding_container>{}, shape, shared_factories...);

  // cast the discarding_container to void
  return new_executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_async_execute_with_shared_inits_returning_void()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape,
                    Factories... shared_factories)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_with_shared_inits_returning_void<
    Executor,
    Function,
    Factories...
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_with_shared_inits_returning_void(check_for_member_function(), ex, f, shape, shared_factories...);
} // end new_executor_traits::async_execute()


} // end agency

