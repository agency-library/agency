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
namespace multi_agent_async_execute_returning_user_specified_container_implementation_strategies
{


template<class Executor, class Function, class Factory>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_async_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.async_execute(f, result_factory, shape);
} // end multi_agent_async_execute_returning_user_specified_container()


template<class Executor, class Function, class Factory>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  multi_agent_async_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, Factory result_factory, typename new_executor_traits<Executor>::shape_type shape)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::then_execute(ex, f, result_factory, shape, ready);
} // end multi_agent_async_execute_returning_user_specified_container()


} // end multi_agent_async_execute_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Factory>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Factory(typename new_executor_traits<Executor>::shape_type)>::type
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    Factory result_factory,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_user_specified_container<
    Executor,
    Function,
    Factory
  >;

  namespace ns = detail::new_executor_traits_detail::multi_agent_async_execute_returning_user_specified_container_implementation_strategies;

  return ns::multi_agent_async_execute_returning_user_specified_container(check_for_member_function(), ex, f, result_factory, shape);
} // end new_executor_traits::async_execute()


} // end agency

