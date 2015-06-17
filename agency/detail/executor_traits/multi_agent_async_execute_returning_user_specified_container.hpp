#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Container, class Executor, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template async_execute<Container>(f, shape);
} // end multi_agent_async_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::template then_execute<Container>(ex, f, shape, ready);
} // end multi_agent_async_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
struct has_multi_agent_async_execute_returning_user_specified_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().template async_execute<Container>(
               std::declval<Function>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,Container>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Container, class Executor, class Function>
using has_multi_agent_async_execute_returning_user_specified_container = typename has_multi_agent_async_execute_returning_user_specified_container_impl<Container,Executor,Function>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function>
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_user_specified_container<
    Container,
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::async_execute()


} // end agency

