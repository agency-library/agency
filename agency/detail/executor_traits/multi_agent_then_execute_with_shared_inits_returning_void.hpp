#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <agency/detail/executor_traits/invoke_and_return_empty.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


__agency_hd_warning_disable__
template<class Executor, class Function, class Future, class... Factories>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_with_shared_inits_returning_void(std::true_type,
                                                            Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  return ex.then_execute(f, shape, fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_void()


template<class Executor, class Function, class Future, class... Factories>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_with_shared_inits_returning_void(std::false_type,
                                                            Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = new_executor_traits<Executor>::template then_execute<discarding_container>(ex, invoke_and_return_empty<Function>{f}, shape, fut, shared_factories...);

  // cast the discarding_container to void
  return new_executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_then_execute_with_shared_inits_returning_void()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future, class... Factories,
           class Enable1,
           class Enable2,
           class Enable3,
           class Enable4
          >
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut,
                   Factories... shared_factories)
{
  namespace ns = detail::new_executor_traits_detail;

  using implementation_strategy = ns::has_multi_agent_then_execute_with_shared_inits_returning_void<
    Executor,
    Function,
    Future,
    Factories...
  >;

  return ns::multi_agent_then_execute_with_shared_inits_returning_void(implementation_strategy(), ex, f, shape, fut, shared_factories...);
} // end new_executor_traits::then_execute()


} // end agency


