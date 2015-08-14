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


template<class Container, class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_with_shared_inits_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return ex.template async_execute<Container>(f, shape, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_user_specified_container()


template<class Container, class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_with_shared_inits_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::template then_execute<Container>(ex, f, shape, ready, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class... Factories,
           class Enable>
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape,
                    Factories... shared_factories)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container<
    Container,
    Executor,
    Function,
    Factories...
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_with_shared_inits_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape, shared_factories...);
} // end new_executor_traits::async_execute()


} // end agency

