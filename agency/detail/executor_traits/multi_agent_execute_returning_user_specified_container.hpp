#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/ignore_tail_parameters_and_invoke.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


__agency_hd_warning_disable__
template<class Container, class Executor, class Function>
__AGENCY_ANNOTATION
Container multi_agent_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template execute<Container>(f, shape);
} // end multi_agent_execute_returning_user_specified_container()




template<class Container, size_t... Indices, class Executor, class Function, class Tuple>
__AGENCY_ANNOTATION
Container multi_agent_execute_returning_user_specified_container_impl(detail::index_sequence<Indices...>,
                                                                      Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                      const Tuple& tuple_of_unit_factories)
{
  return new_executor_traits<Executor>::template execute<Container>(ex, ignore_tail_parameters_and_invoke<Function>{f}, shape, std::get<Indices>(tuple_of_unit_factories)...);
} // end multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
__AGENCY_ANNOTATION
Container multi_agent_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto tuple_of_unit_factories = new_executor_traits_detail::make_tuple_of_unit_factories(ex);

  return multi_agent_execute_returning_user_specified_container_impl<Container>(detail::make_index_sequence<std::tuple_size<decltype(tuple_of_unit_factories)>::value>(), ex, f, shape, tuple_of_unit_factories);
} // end multi_agent_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function>
__AGENCY_ANNOTATION
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

