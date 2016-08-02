#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/execution/executor/detail/executor_traits/discarding_container.hpp>
#include <agency/execution/executor/detail/executor_traits/container_factory.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


__agency_exec_check_disable__
template<class Executor, class Function>
__AGENCY_ANNOTATION
void multi_agent_execute_returning_void(std::true_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, shape);
} // end multi_agent_execute_returning_void()


template<class Executor, class Function>
__AGENCY_ANNOTATION
void multi_agent_execute_returning_void(std::false_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape)
{
  auto g = [=](const typename executor_traits<Executor>::index_type& idx) mutable
  {
    agency::detail::invoke(f, idx);

    // return something which can be cheaply discarded
    return 0;
  };

  executor_traits<Executor>::execute(ex, g, container_factory<discarding_container>{}, shape);
} // end multi_agent_execute_returning_void()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
__AGENCY_ANNOTATION
void executor_traits<Executor>
  ::execute(typename executor_traits<Executor>::executor_type& ex,
            Function f,
            typename executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::executor_traits_detail::has_multi_agent_execute_returning_void<
    Executor,
    Function
  >;

  return detail::executor_traits_detail::multi_agent_execute_returning_void(check_for_member_function(), ex, f, shape);
} // end executor_traits::execute()


} // end agency

