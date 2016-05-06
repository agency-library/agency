#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/executor/detail/executor_traits/discarding_container.hpp>
#include <agency/executor/detail/executor_traits/container_factory.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


__agency_exec_check_disable__
template<class Executor, class Function, class... Factories>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_async_execute_with_shared_inits_returning_void(std::true_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return ex.async_execute(f, shape, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_void()


template<class Executor, class Function, class... Factories>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_async_execute_with_shared_inits_returning_void(std::false_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = executor_traits<Executor>::async_execute(ex, invoke_and_return_empty<Function>{f}, container_factory<discarding_container>{}, shape, shared_factories...);

  // cast the discarding_container to void
  return executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_async_execute_with_shared_inits_returning_void()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  executor_traits<Executor>
    ::async_execute(typename executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename executor_traits<Executor>::shape_type shape,
                    Factories... shared_factories)
{
  using check_for_member_function = detail::executor_traits_detail::has_multi_agent_async_execute_with_shared_inits_returning_void<
    Executor,
    Function,
    Factories...
  >;

  return detail::executor_traits_detail::multi_agent_async_execute_with_shared_inits_returning_void(check_for_member_function(), ex, f, shape, shared_factories...);
} // end executor_traits::async_execute()


} // end agency

