#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/terminal_multi_agent_execute_returning_user_specified_container.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_then_execute_returning_user_specified_container_implementation_strategies
{


struct use_multi_agent_then_execute_returning_user_specified_container_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_single_agent_then_execute_with_nested_terminal_multi_agent_execute {};


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
using has_multi_agent_when_all_execute_and_select =
  new_executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    multi_agent_then_execute_returning_user_specified_container_functor<Function>,
    detail::tuple<
      typename new_executor_traits<Executor>::template future<Container>,
      Future
    >,
    0
  >;


template<class Container, class Executor, class Function, class Future>
using select_multi_agent_then_execute_returning_user_specified_container_implementation =
  typename std::conditional<
    has_multi_agent_then_execute_returning_user_specified_container<Container,Executor,Function,Future>::value,
    use_multi_agent_then_execute_returning_user_specified_container_member_function,
    typename std::conditional<
      has_multi_agent_when_all_execute_and_select<Container,Executor,Function,Future>::value,
      use_multi_agent_when_all_execute_and_select_member_function,
      use_single_agent_then_execute_with_nested_terminal_multi_agent_execute
    >::type
  >::type;


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(use_multi_agent_then_execute_returning_user_specified_container_member_function,
                                                              Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.template then_execute<Container>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_user_specified_container()


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_member_function,
                                                              Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  using traits = new_executor_traits<Executor>;

  auto results = traits::template make_ready_future<Container>(ex, shape);

  auto results_and_fut = detail::make_tuple(std::move(results), std::move(fut));

  return ex.template when_all_execute_and_select<0>(multi_agent_then_execute_returning_user_specified_container_functor<Function>{f}, shape, results_and_fut);
} // end multi_agent_then_execute_returning_user_specified_container()


// overload for non-void futures
template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container_impl(Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut,
                                                                   typename std::enable_if<
                                                                     !std::is_void<
                                                                       typename future_traits<Future>::value_type
                                                                     >::value
                                                                   >::type* = 0)
{
  using value_type = typename future_traits<Future>::value_type;

  return new_executor_traits<Executor>::then_execute(ex, [=, &ex](value_type& val)
  {
    using index_type = typename new_executor_traits<Executor>::index_type;

    return new_executor_traits_detail::terminal_multi_agent_execute_returning_user_specified_container<Container>(ex, [=,&val](const index_type& idx)
    {
      return f(idx, val);
    },
    shape);
  },
  fut);
} // end multi_agent_then_execute_returning_user_specified_container_impl()


// overload for void futures
template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container_impl(Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut,
                                                                   typename std::enable_if<
                                                                     std::is_void<
                                                                       typename future_traits<Future>::value_type
                                                                     >::value
                                                                   >::type* = 0)
{
  return new_executor_traits<Executor>::then_execute(ex, [=, &ex]
  {
    using index_type = typename new_executor_traits<Executor>::index_type;

    return new_executor_traits_detail::terminal_multi_agent_execute_returning_user_specified_container<Container>(ex, [=](const index_type& idx)
    {
      return f(idx);
    },
    shape);
  },
  fut);
} // end multi_agent_then_execute_returning_user_specified_container_impl()


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(use_single_agent_then_execute_with_nested_terminal_multi_agent_execute,
                                                              Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return multi_agent_then_execute_returning_user_specified_container_implementation_strategies::multi_agent_then_execute_returning_user_specified_container_impl<Container>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_user_specified_container()


} // end multi_agent_then_execute_returning_user_specified_container_implementation_strategies
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
  namespace ns = detail::new_executor_traits_detail::multi_agent_then_execute_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_then_execute_returning_user_specified_container_implementation<Container,Executor,Function,Future>;

  return ns::multi_agent_then_execute_returning_user_specified_container<Container>(implementation_strategy(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency

