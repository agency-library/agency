#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(typename new_executor_traits<Executor>::index_type)
  >::type
>
  multi_agent_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, shape);
} // end multi_agent_execute_returning_default_container()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(typename new_executor_traits<Executor>::index_type)
  >::type
>
  multi_agent_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  return new_executor_traits<Executor>::new_execute(ex, f, container_factory<container_type>{}, shape);
} // end multi_agent_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(typename new_executor_traits<Executor>::index_type)
  >::type
>
  new_executor_traits<Executor>
    ::execute(typename new_executor_traits<Executor>::executor_type& ex,
              Function f,
              typename new_executor_traits<Executor>::shape_type shape)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_execute_returning_user_specified_container<
    Executor,
    Function,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_execute_returning_default_container(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::execute()


} // end agency

