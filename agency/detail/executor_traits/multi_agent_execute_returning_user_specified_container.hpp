#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/ignore_tail_parameters_and_invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace multi_agent_execute_returning_user_specified_container_implementation_strategies
{


__agency_exec_check_disable__
template<class Executor, class Function, class Factory>
__AGENCY_ANNOTATION
result_of_t<Factory(typename executor_traits<Executor>::shape_type)>
  multi_agent_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, Factory result_factory, typename executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, result_factory, shape);
} // end multi_agent_execute_returning_user_specified_container()




template<size_t... Indices, class Executor, class Function, class Factory, class Tuple>
__AGENCY_ANNOTATION
result_of_t<Factory(typename executor_traits<Executor>::shape_type)>
  multi_agent_execute_returning_user_specified_container_impl(detail::index_sequence<Indices...>,
                                                              Executor& ex, Function f, Factory result_factory, typename executor_traits<Executor>::shape_type shape,
                                                              const Tuple& tuple_of_unit_factories)
{
  return executor_traits<Executor>::execute(ex, ignore_tail_parameters_and_invoke<Function>{f}, result_factory, shape, std::get<Indices>(tuple_of_unit_factories)...);
} // end multi_agent_execute_returning_user_specified_container()


template<class Executor, class Function, class Factory>
__AGENCY_ANNOTATION
result_of_t<Factory(typename executor_traits<Executor>::shape_type)>
  multi_agent_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, Factory result_factory, typename executor_traits<Executor>::shape_type shape)
{
  auto tuple_of_unit_factories = executor_traits_detail::make_tuple_of_unit_factories(ex);

  return multi_agent_execute_returning_user_specified_container_impl(detail::make_index_sequence<std::tuple_size<decltype(tuple_of_unit_factories)>::value>(), ex, f, result_factory, shape, tuple_of_unit_factories);
} // end multi_agent_execute_returning_user_specified_container()


} // end multi_agent_execute_returning_user_specified_container_implementation_strategies
} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Factory>
__AGENCY_ANNOTATION
detail::result_of_t<Factory(typename executor_traits<Executor>::shape_type)> executor_traits<Executor>
  ::execute(typename executor_traits<Executor>::executor_type& ex,
            Function f,
            Factory result_factory,
            typename executor_traits<Executor>::shape_type shape)
{
  namespace ns = detail::executor_traits_detail::multi_agent_execute_returning_user_specified_container_implementation_strategies;

  using check_for_member_function = agency::detail::executor_traits_detail::has_multi_agent_execute_returning_user_specified_container<
    Executor,
    Function,
    Factory
  >;

  return ns::multi_agent_execute_returning_user_specified_container(check_for_member_function(), ex, f, result_factory, shape);
} // end executor_traits::execute()


} // end agency

