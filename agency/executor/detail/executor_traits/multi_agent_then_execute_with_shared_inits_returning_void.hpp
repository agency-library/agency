#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/executor/detail/executor_traits/discarding_container.hpp>
#include <agency/executor/detail/executor_traits/container_factory.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


__agency_exec_check_disable__
template<class Executor, class Function, class Future, class... Factories>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_then_execute_with_shared_inits_returning_void(std::true_type,
                                                            Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  return ex.then_execute(f, shape, fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_void()


template<class Executor, class Function, class Future, class... Factories>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_then_execute_with_shared_inits_returning_void(std::false_type,
                                                            Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = executor_traits<Executor>::then_execute(ex, invoke_and_return_empty<Function>{f}, container_factory<discarding_container>{}, shape, fut, shared_factories...);

  // cast the discarding_container to void
  return executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_then_execute_with_shared_inits_returning_void()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future, class... Factories,
           class Enable1,
           class Enable2,
           class Enable3,
           class Enable4
          >
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  executor_traits<Executor>
    ::then_execute(typename executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename executor_traits<Executor>::shape_type shape,
                   Future& fut,
                   Factories... shared_factories)
{
  namespace ns = detail::executor_traits_detail;

  using implementation_strategy = ns::has_multi_agent_then_execute_with_shared_inits_returning_void<
    Executor,
    Function,
    Future,
    Factories...
  >;

  return ns::multi_agent_then_execute_with_shared_inits_returning_void(implementation_strategy(), ex, f, shape, fut, shared_factories...);
} // end executor_traits::then_execute()


} // end agency


