#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <type_traits>


// this is a version of multi-agent execute() returning void which does not attempt to recurse to async_execute() + wait()


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace terminal_multi_agent_execute_returning_void_implementation_strategies
{


struct use_multi_agent_execute_returning_void_member_function {};

struct use_multi_agent_execute_returning_discarding_container_member_function {};

struct use_for_loop {};


struct empty {};


template<class Function>
struct invoke_and_return_empty
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  empty operator()(const Index& idx, Args&... args) const
  {
    f(idx, args...);

    // return something which can be cheaply discarded
    return empty();
  }
};


template<class Executor, class Function>
using has_multi_agent_execute_returning_discarding_container = 
  has_multi_agent_execute_returning_user_specified_container<
    discarding_container,
    Executor,
    invoke_and_return_empty<Function>
  >;


template<class Executor, class Function>
using select_multi_agent_terminal_execute_returning_void_implementation = 
  typename std::conditional<
    has_multi_agent_execute_returning_void<Executor,Function>::value,
    use_multi_agent_execute_returning_void_member_function,
    typename std::conditional<
      has_multi_agent_execute_returning_discarding_container<Executor,Function>::value,
      use_multi_agent_execute_returning_discarding_container_member_function,
      use_for_loop
    >::type
  >::type;


template<class Executor, class Function>
void terminal_multi_agent_execute_returning_void(use_multi_agent_execute_returning_void_member_function,
                                                 Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, shape);
} // end terminal_multi_agent_execute_returning_void()


template<class Executor, class Function>
void terminal_multi_agent_execute_returning_void(use_multi_agent_execute_returning_discarding_container_member_function,
                                                 Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  ex.template execute<discarding_container>(invoke_and_return_empty<Function>{f}, shape);
} // end terminal_multi_agent_execute_returning_void()


template<class Executor, class Function>
void terminal_multi_agent_execute_returning_void(use_for_loop,
                                                 Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using index_type = typename new_executor_traits<Executor>::index_type;

  // XXX generalize to multidimensions or just use sequential_executor
  for(index_type idx = 0; idx < shape; ++idx)
  {
    f(idx);
  }
} // end multi_agent_execute_returning_void()


} // end terminal_multi_agent_execute_returning_void_implementation_strategies


template<class Executor, class Function>
void terminal_multi_agent_execute_returning_void(Executor& ex,
                                                 Function f,
                                                 typename new_executor_traits<Executor>::shape_type shape)
{
  namespace ns = detail::new_executor_traits_detail::terminal_multi_agent_execute_returning_void_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_terminal_execute_returning_void_implementation<
    Executor,
    Function
  >;

  return ns::terminal_multi_agent_execute_returning_void(implementation_strategy(), ex, f, shape);
} // end terminal_multi_agent_execute_returning_void()


} // end new_executor_traits_detail
} // end detail
} // end agency


