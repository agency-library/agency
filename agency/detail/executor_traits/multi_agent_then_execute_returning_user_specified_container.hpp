#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/ignore_tail_parameters_and_invoke.hpp>
#include <agency/functional.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_then_execute_returning_user_specified_container_implementation_strategies
{


struct use_multi_agent_then_execute_returning_user_specified_container_member_function {};

struct use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container {};

template<class Executor, class Function, class Factory, class Future>
using select_multi_agent_then_execute_returning_user_specified_container_implementation =
  typename std::conditional<
    has_multi_agent_then_execute_returning_user_specified_container<Executor,Function,Factory,Future>::value,
    use_multi_agent_then_execute_returning_user_specified_container_member_function,
    use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container
  >::type;


__agency_hd_warning_disable__
template<class Executor, class Function, class Factory, class Future>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_then_execute_returning_user_specified_container(use_multi_agent_then_execute_returning_user_specified_container_member_function,
                                                              Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, result_factory, shape, fut);
} // end multi_agent_then_execute_returning_user_specified_container()


template<class Function, class T>
struct ignore_tail_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  typename std::result_of<Function(Index,T&)>::type
  operator()(const Index& idx, T& past_arg, Args&&...) const
  {
    return agency::invoke(f, idx, past_arg);
  }
};


template<class Function>
struct ignore_tail_parameters_and_invoke<Function,void>
{
  mutable Function f;

  __agency_hd_warning_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  typename std::result_of<Function(Index)>::type
  operator()(const Index& idx, Args&&...) const
  {
    return agency::invoke(f, idx);
  }
};


template<size_t... Indices, class Executor, class Function, class Factory, class Future, class Tuple>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_then_execute_returning_user_specified_container_impl(detail::index_sequence<Indices...>,
                                                                   Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape, Future& fut,
                                                                   const Tuple& tuple_of_unit_factories)
{
  using value_type = typename future_traits<Future>::value_type;

  return new_executor_traits<Executor>::then_execute(ex, ignore_tail_parameters_and_invoke<Function,value_type>{f}, result_factory, shape, fut, std::get<Indices>(tuple_of_unit_factories)...);
} // end multi_agent_then_execute_returning_user_specified_container_impl()


template<class Executor, class Function, class Factory, class Future>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_then_execute_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container,
                                                              Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  auto tuple_of_unit_factories = new_executor_traits_detail::make_tuple_of_unit_factories(ex);

  return multi_agent_then_execute_returning_user_specified_container_impl(detail::make_index_sequence<std::tuple_size<decltype(tuple_of_unit_factories)>::value>(), ex, f, result_factory, shape, fut, tuple_of_unit_factories);
} // end multi_agent_then_execute_returning_user_specified_container()


} // end multi_agent_then_execute_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future, class Factory,
           class Enable1,
           class Enable2
           >
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   Factory result_factory,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_then_execute_returning_user_specified_container_implementation_strategies;

  using check_for_member_function = ns::select_multi_agent_then_execute_returning_user_specified_container_implementation<Executor,Function,Factory,Future>;

  return ns::multi_agent_then_execute_returning_user_specified_container(check_for_member_function(), ex, f, result_factory, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency

