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


template<class Executor, class Function, class... Factories>
using select_multi_agent_execute_with_shared_inits_returning_default_container_implementation = 
  typename std::conditional<
    has_multi_agent_execute_with_shared_inits_returning_default_container<
      Executor, Function, Factories...
    >::value,
    use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
    use_multi_agent_execute_with_shared_inits_returning_user_specified_container
  >::type;


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(
      typename new_executor_traits<Executor>::index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
                                                                    Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                    Factories... shared_factories)
{
  return ex.execute(f, shape, shared_factories...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(
      typename new_executor_traits<Executor>::index_type,
      typename std::result_of<Factories()>::type&...
    )
  >::type
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container,
                                                                    Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape,
                                                                    Factories... shared_factories)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(
        typename new_executor_traits<Executor>::index_type,
        typename std::result_of<Factories()>::type&...
      )
    >::type
  >;

  return new_executor_traits<Executor>::template execute<container_type>(ex, f, shape, shared_factories...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


} // end multi_execute_with_shared_inits_returning_default_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
typename new_executor_traits<Executor>::template container<
  typename std::result_of<
    Function(typename new_executor_traits<Executor>::index_type, typename std::result_of<Factories()>::type&...)
  >::type
>
  new_executor_traits<Executor>
    ::execute(typename new_executor_traits<Executor>::executor_type& ex,
              Function f,
              typename new_executor_traits<Executor>::shape_type shape,
              Factories... shared_factories)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_execute_with_shared_inits_returning_default_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_default_container_implementation<
    Executor,
    Function,
    Factories...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_default_container(implementation_strategy(), ex, f, shape, shared_factories...);
} // end new_executor_traits::execute()


} // end agency

