#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.template then_execute<Container>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_user_specified_container()


template<class Function>
struct multi_agent_then_execute_returning_user_specified_container_functor
{
  mutable Function f;

  template<class Index, class Container, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c, Arg& arg) const
  {
    c[idx] = f(idx, arg);
  }

  template<class Index, class Container>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c) const
  {
    c[idx] = f(idx);
  }
};


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  using traits = new_executor_traits<Executor>;

  auto results = traits::template make_ready_future<Container>(ex, shape);

  auto results_and_fut = detail::make_tuple(std::move(results), std::move(fut));

  return traits::template when_all_execute_and_select<0>(ex, multi_agent_then_execute_returning_user_specified_container_functor<Function>{f}, shape, results_and_fut);
} // end multi_agent_then_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
           >
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_user_specified_container<
    Container,
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency

