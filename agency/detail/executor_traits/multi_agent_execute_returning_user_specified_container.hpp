#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Container, class Executor, class Function>
Container multi_agent_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template execute<Container>(f, shape);
} // end multi_agent_execute_returning_user_specified_container()


template<class Function>
struct ignore_tail_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  typename std::result_of<Function(Index)>::type
  operator()(const Index& idx, Args&&...) const
  {
    // XXX should use std::invoke
    return f(idx);
  }
};


template<class Container, size_t... Indices, class Executor, class Function, class Tuple>
Container multi_agent_execute_returning_user_specified_container_impl(detail::index_sequence<Indices...>,
                                                                      Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                      const Tuple& tuple_of_ignored_parameters)
{
  return new_executor_traits<Executor>::template execute<Container>(ex, ignore_tail_parameters_and_invoke<Function>{f}, shape, std::get<Indices>(tuple_of_ignored_parameters)...);
} // end multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container multi_agent_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  constexpr size_t num_ignored_parameters = new_executor_traits<Executor>::execution_depth;

  auto tuple_of_ignored_parameters = detail::make_homogeneous_tuple<num_ignored_parameters>(detail::ignore);

  return multi_agent_execute_returning_user_specified_container_impl<Container>(detail::make_index_sequence<num_ignored_parameters>(), ex, f, shape, tuple_of_ignored_parameters);
} // end multi_agent_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function>
Container new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_execute_returning_user_specified_container<
    Container,
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::multi_agent_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::execute()


} // end agency

