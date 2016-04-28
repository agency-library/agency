#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Function>
typename executor_traits<Executor>::template container<
  result_of_t<
    Function(typename executor_traits<Executor>::index_type)
  >
>
  multi_agent_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, shape);
} // end multi_agent_execute_returning_default_container()


template<class Executor, class Function>
typename executor_traits<Executor>::template container<
  result_of_t<
    Function(typename executor_traits<Executor>::index_type)
  >
>
  multi_agent_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  using container_type = typename executor_traits<Executor>::template container<
    result_of_t<
      Function(typename executor_traits<Executor>::index_type)
    >
  >;

  return executor_traits<Executor>::execute(ex, f, container_factory<container_type>{}, shape);
} // end multi_agent_execute_returning_user_specified_container()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename executor_traits<Executor>::template container<
  detail::result_of_t<
    Function(typename executor_traits<Executor>::index_type)
  >
>
  executor_traits<Executor>
    ::execute(typename executor_traits<Executor>::executor_type& ex,
              Function f,
              typename executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::executor_traits_detail::has_multi_agent_execute_returning_default_container<
    Executor,
    Function
  >;

  return detail::executor_traits_detail::multi_agent_execute_returning_default_container(check_for_member_function(), ex, f, shape);
} // end executor_traits::execute()


} // end agency

