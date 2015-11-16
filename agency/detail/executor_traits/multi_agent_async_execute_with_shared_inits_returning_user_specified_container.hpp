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
namespace multi_agent_async_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
{


template<class Executor, class Function, class Factory, class... Factories>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_async_execute_with_shared_inits_returning_user_specified_container(std::true_type, Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return ex.async_execute(f, result_factory, shape, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_user_specified_container()


template<class Executor, class Function, class Factory, class... Factories>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_async_execute_with_shared_inits_returning_user_specified_container(std::false_type, Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::then_execute(ex, f, result_factory, shape, ready, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_user_specified_container()


} // end multi_agent_async_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Factory, class... Factories,
           class Enable>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    Factory result_factory,
                    typename new_executor_traits<Executor>::shape_type shape,
                    Factories... shared_factories)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container<
    Executor,
    Function,
    Factory,
    Factories...
  >;

  namespace ns = detail::new_executor_traits_detail::multi_agent_async_execute_with_shared_inits_returning_user_specified_container_implementation_strategies;

  return ns::multi_agent_async_execute_with_shared_inits_returning_user_specified_container(check_for_member_function(), ex, f, result_factory, shape, shared_factories...);
} // end new_executor_traits::async_execute()


} // end agency

