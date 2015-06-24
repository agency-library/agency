#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/terminal_multi_agent_execute_returning_void.hpp>
#include <agency/detail/shape_cast.hpp>
#include <type_traits>
#include <utility>


// this is a version of single-agent execute() which does not attempt to recurse to async_execute() + wait()


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace terminal_single_agent_execute_implementation_strategies
{

struct use_single_agent_execute_member_function {};

struct use_terminal_multi_agent_execute_returning_void {};


template<class Executor, class Function>
using select_multi_agent_terminal_execute_returning_void_implementation = 
  typename std::conditional<
    has_single_agent_execute<Executor,Function>::value,
    use_single_agent_execute_member_function,
    use_terminal_multi_agent_execute_returning_void
  >::type;


template<class Executor, class Function>
typename std::result_of<Function()>::type
  terminal_single_agent_execute(use_single_agent_execute_member_function,
                                Executor& ex, Function&& f)
{
  return ex.execute(std::forward<Function>(f));
} // end terminal_single_agent_execute()


template<class Executor, class Function>
typename std::result_of<Function()>::type
  terminal_single_agent_execute(use_terminal_multi_agent_execute_returning_void,
                                Executor& ex, Function&& f,
                                typename std::enable_if<
                                  !std::is_void<
                                    typename std::result_of<Function()>::type
                                  >::value
                                >::type* = 0)
{
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  using index_type = typename new_executor_traits<Executor>::index_type;

  // XXX should uninitialize this and call placement new inside the lambda below
  typename std::result_of<Function()>::type result;

  new_executor_traits_detail::terminal_multi_agent_execute_returning_void(ex, [&](const index_type&)
  {
    // XXX should use std::invoke(f)
    result = f();
  },
  detail::shape_cast<shape_type>(1));

  return result;
} // end terminal_single_agent_execute()


template<class Executor, class Function>
typename std::result_of<Function()>::type
  terminal_single_agent_execute(use_terminal_multi_agent_execute_returning_void,
                                Executor& ex, Function&& f,
                                typename std::enable_if<
                                  std::is_void<
                                    typename std::result_of<Function()>::type
                                  >::value
                                >::type* = 0)
{
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  using index_type = typename new_executor_traits<Executor>::index_type;

  new_executor_traits_detail::terminal_multi_agent_execute_returning_void(ex, [&](const index_type&)
  {
    // XXX should use std::invoke(f)
    f();
  },
  detail::shape_cast<shape_type>(1));
} // end terminal_single_agent_execute()


} // end terminal_multi_agent_execute_returning_void_implementation_strategies


template<class Executor, class Function>
typename std::result_of<Function()>::type
  terminal_single_agent_execute(Executor& ex,
                                Function&& f)
{
  namespace ns = detail::new_executor_traits_detail::terminal_single_agent_execute_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_terminal_execute_returning_void_implementation<
    Executor,
    Function
  >;

  return ns::terminal_single_agent_execute(implementation_strategy(), ex, f);
} // end terminal_single_agent_execute()


} // end new_executor_traits_detail
} // end detail
} // end agency



