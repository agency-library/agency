#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
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
typename executor_traits<Executor>::template container<
  result_of_t<
    Function(
      typename executor_traits<Executor>::index_type,
      result_of_t<Factories()>&...
    )
  >
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_default_container_member_function,
                                                                    Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape,
                                                                    Factories... shared_factories)
{
  return ex.execute(f, shape, shared_factories...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


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
typename executor_traits<Executor>::template container<
  result_of_t<
    Function(
      typename executor_traits<Executor>::index_type,
      result_of_t<Factories()>&...
    )
  >
>
  multi_agent_execute_with_shared_inits_returning_default_container(use_multi_agent_execute_with_shared_inits_returning_user_specified_container,
                                                                    Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape,
                                                                    Factories... shared_factories)
{
  using container_type = typename executor_traits<Executor>::template container<
    result_of_t<
      Function(
        typename executor_traits<Executor>::index_type,
        result_of_t<Factories()>&...
      )
    >
  >;

  return executor_traits<Executor>::execute(ex, f, factory<container_type>{}, shape, shared_factories...);
} // end multi_agent_execute_with_shared_inits_returning_default_container()


} // end multi_execute_with_shared_inits_returning_default_container_implementation_strategies
} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
typename executor_traits<Executor>::template container<
  detail::result_of_t<
    Function(typename executor_traits<Executor>::index_type, detail::result_of_t<Factories()>&...)
  >
>
  executor_traits<Executor>
    ::execute(typename executor_traits<Executor>::executor_type& ex,
              Function f,
              typename executor_traits<Executor>::shape_type shape,
              Factories... shared_factories)
{
  namespace ns = detail::executor_traits_detail::multi_agent_execute_with_shared_inits_returning_default_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_execute_with_shared_inits_returning_default_container_implementation<
    Executor,
    Function,
    Factories...
  >;

  return ns::multi_agent_execute_with_shared_inits_returning_default_container(implementation_strategy(), ex, f, shape, shared_factories...);
} // end executor_traits::execute()


} // end agency

