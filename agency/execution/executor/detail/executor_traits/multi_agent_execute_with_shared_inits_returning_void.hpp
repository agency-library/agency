#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/execution/executor/detail/executor_traits/discarding_container.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace multi_agent_execute_with_shared_inits_returning_void_implementation_strategies
{


struct use_multi_agent_execute_with_shared_inits_returning_void_member_function {};

struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container {};


template<class Executor, class Function, class... Factories>
using select_multi_agent_execute_with_shared_inits_returning_void_implementation =
  typename std::conditional<
    has_multi_agent_execute_with_shared_inits_returning_void<
      Executor, Function, Factories...
    >::value,
    use_multi_agent_execute_with_shared_inits_returning_void_member_function,
    use_multi_agent_execute_with_shared_inits_returning_user_specified_container
  >::type;


__agency_exec_check_disable__
template<class Executor, class Function, class... Factories>
__AGENCY_ANNOTATION
void multi_agent_execute_with_shared_inits_returning_void(use_multi_agent_execute_with_shared_inits_returning_void_member_function,
                                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape,
                                                          Factories... shared_factories)
{
  return ex.execute(f, shape, shared_factories...);
} // end multi_agent_execute_with_shared_inits_returning_void()


template<class Function>
struct multi_agent_execute_with_shared_inits_returning_void_functor
{
  struct empty {};

  Function f;

  template<class... Args>
  __AGENCY_ANNOTATION
  empty operator()(Args&&... args)
  {
    agency::detail::invoke(f, std::forward<Args>(args)...);

    // return something which can be cheaply discarded
    return empty();
  }
};


template<class Result>
struct factory
{
  template<class... Args>
  __AGENCY_ANNOTATION
  Result operator()(Args&&... args) const
  {
    return Result(std::forward<Args>(args)...);
  }
};


template<class Executor, class Function, class... Factories>
__AGENCY_ANNOTATION
void multi_agent_execute_with_shared_inits_returning_void(use_multi_agent_execute_with_shared_inits_returning_user_specified_container,
                                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape,
                                                          Factories... shared_factories)
{
  auto g = multi_agent_execute_with_shared_inits_returning_void_functor<Function>{f};

  executor_traits<Executor>::execute(ex, g, factory<discarding_container>{}, shape, shared_factories...);
} // end multi_agent_execute_returning_void()


} // end multi_agent_execute_with_shared_inits_returning_void_implementation_strategies
} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
__AGENCY_ANNOTATION
void executor_traits<Executor>
  ::execute(typename executor_traits<Executor>::executor_type& ex,
            Function f,
            typename executor_traits<Executor>::shape_type shape,
            Factories... shared_factories)
{
  namespace ns = detail::executor_traits_detail::multi_agent_execute_with_shared_inits_returning_void_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_void_implementation<
    Executor,
    Function,
    Factories...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_void(implementation_strategy(), ex, f, shape, shared_factories...);
} // end executor_traits::execute()


} // end agency

