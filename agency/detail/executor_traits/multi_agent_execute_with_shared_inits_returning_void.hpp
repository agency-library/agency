#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <agency/functional.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_execute_with_shared_inits_returning_void_implementation_strategies
{


struct use_multi_agent_execute_with_shared_inits_returning_void_member_function {};

struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container {};


template<class Executor, class Function, class... Types>
using select_multi_agent_execute_with_shared_inits_returning_void_implementation =
  typename std::conditional<
    has_multi_agent_execute_with_shared_inits_returning_void<
      Executor, Function, Types...
    >::value,
    use_multi_agent_execute_with_shared_inits_returning_void_member_function,
    use_multi_agent_execute_with_shared_inits_returning_user_specified_container
  >::type;


template<class Executor, class Function, class... Types>
void multi_agent_execute_with_shared_inits_returning_void(use_multi_agent_execute_with_shared_inits_returning_void_member_function,
                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                          Types&&... shared_inits)
{
  return ex.execute(f, shape, std::forward<Types>(shared_inits)...);
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
    agency::invoke(f, std::forward<Args>(args)...);

    // return something which can be cheaply discarded
    return empty();
  }
};


template<class Executor, class Function, class... Types>
void multi_agent_execute_with_shared_inits_returning_void(use_multi_agent_execute_with_shared_inits_returning_user_specified_container,
                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                          Types&&... shared_inits)
{
  auto g = multi_agent_execute_with_shared_inits_returning_void_functor<Function>{f};

  new_executor_traits<Executor>::template execute<discarding_container>(ex, g, shape, std::forward<Types>(shared_inits)...);
} // end multi_agent_execute_returning_void()


} // end multi_agent_execute_with_shared_inits_returning_void_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Types,
           class Enable1,
           class Enable2>
void new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            typename new_executor_traits<Executor>::shape_type shape,
            Types&&... shared_inits)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_execute_with_shared_inits_returning_void_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_void_implementation<
    Executor,
    Function,
    Types&&...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_void(implementation_strategy(), ex, f, shape, std::forward<Types>(shared_inits)...);
} // end new_executor_traits::execute()


} // end agency

