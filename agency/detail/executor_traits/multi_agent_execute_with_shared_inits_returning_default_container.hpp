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
namespace multi_agent_execute_with_shared_inits_returning_default_container_implementation_strategies
{


struct use_multi_agent_execute_with_shared_inits_returning_default_container_member_function {};

struct use_multi_agent_execute_with_shared_inits_returning_user_specified_container {};


template<class Executor, class Function, class... Types>
using select_multi_agent_execute_with_shared_inits_returning_default_container_implementation = 
  typename std::conditional<
    has_multi_agent_execute_with_shared_inits_returning_default_container<
      Executor, Function, Types...
    >::value,
    use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
    use_multi_agent_execute_with_shared_inits_returning_user_specified_container
  >::type;


template<class Executor, class Function, class... Types>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(
      typename new_executor_traits<Executor>::index_type,
      typename std::decay<Types>::type&...
    )
  >::type
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
                                                                    Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                    Types&&... shared_inits)
{
  return ex.execute(f, shape, std::forward<Types>(shared_inits)...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


template<class Executor, class Function, class... Types>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(
      typename new_executor_traits<Executor>::index_type,
      typename std::decay<Types>::type&...
    )
  >::type
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container,
                                                                    Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                    Types&&... shared_inits)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(
        typename new_executor_traits<Executor>::index_type,
        typename std::decay<Types>::type&...
      )
    >::type
  >;

  return new_executor_traits<Executor>::template execute<container_type>(ex, f, shape, std::forward<Types>(shared_inits)...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


} // end multi_execute_with_shared_inits_returning_default_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class T1, class... Types,
           class Enable1,
           class Enable2>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(typename new_executor_traits<Executor>::index_type, typename std::decay<T1>::type&, typename std::decay<Types>::type&...)
  >::type
>
  new_executor_traits<Executor>
    ::execute(typename new_executor_traits<Executor>::executor_type& ex,
              Function f,
              typename new_executor_traits<Executor>::shape_type shape,
              T1&& outer_shared_init, Types&&... inner_shared_inits)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_execute_with_shared_inits_returning_default_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_default_container_implementation<
    Executor,
    Function,
    T1&&, Types&&...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_default_container(implementation_strategy(), ex, f, shape, std::forward<T1>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
} // end new_executor_traits::execute()


} // end agency

