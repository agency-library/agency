#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <agency/functional.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace multi_agent_then_execute_returning_void_implementation_strategies
{


struct use_multi_agent_then_execute_returning_void_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_then_execute_returning_discarding_container_and_cast {};


template<class Executor, class Function, class Future>
using has_multi_agent_when_all_execute_and_select = 
  executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    Function,
    detail::tuple<Future>
  >;


template<class Executor, class Function, class Future>
using select_multi_agent_then_execute_returning_void_implementation_strategy =
  typename std::conditional<
    has_multi_agent_then_execute_returning_void<Executor,Function, Future>::value,
    use_multi_agent_then_execute_returning_void_member_function,
    typename std::conditional<
      has_multi_agent_when_all_execute_and_select<Executor,Function,Future>::value,
      use_multi_agent_when_all_execute_and_select_member_function,
      use_multi_agent_then_execute_returning_discarding_container_and_cast
    >::type
  >::type;


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(use_multi_agent_then_execute_returning_void_member_function,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, shape, fut);
} // end multi_agent_then_execute_returning_void()


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(use_multi_agent_when_all_execute_and_select_member_function,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.when_all_execute_and_select(f, shape, detail::make_tuple(std::move(fut)));
} // end multi_agent_then_execute_returning_void()


struct empty {};


template<class Function>
struct invoke_and_return_empty
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  empty operator()(const Index& idx, Args&... args) const
  {
    agency::invoke(f, idx, args...);

    // return something which can be cheaply discarded
    return empty();
  }
};


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(use_multi_agent_then_execute_returning_discarding_container_and_cast,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut)
{
  // invoke f and generate dummy results into a discarding_container
  auto fut2 = executor_traits<Executor>::then_execute(ex, invoke_and_return_empty<Function>{f}, container_factory<discarding_container>{}, shape, fut);

  // cast the discarding_container to void
  return executor_traits<Executor>::template future_cast<void>(ex, fut2);
} // end multi_agent_then_execute_returning_void()


} // end multi_agent_then_execute_returning_void_implementation_strategies
} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
          >
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<void>
  executor_traits<Executor>
    ::then_execute(typename executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  namespace ns = detail::executor_traits_detail::multi_agent_then_execute_returning_void_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_then_execute_returning_void_implementation_strategy<
    Executor,
    Function,
    Future
  >;

  return ns::multi_agent_then_execute_returning_void(implementation_strategy(), ex, f, shape, fut);
} // end executor_traits::then_execute()


} // end agency


