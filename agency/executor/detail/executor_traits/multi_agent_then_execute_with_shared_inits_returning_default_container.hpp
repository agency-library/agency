#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/executor/detail/executor_traits/container_factory.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Function, class Future, class... Factories>
typename executor_traits<Executor>::template future<
  typename executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename executor_traits<Executor>::shape_type,
      Future,
      detail::result_of_t<Factories()>&...
    >
  >
>
  multi_agent_then_execute_with_shared_inits_returning_default_container(std::true_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  return ex.then_execute(f, shape, fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_default_container()


template<class Executor, class Function, class Future, class... Factories>
typename executor_traits<Executor>::template future<
  typename executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename executor_traits<Executor>::shape_type,
      Future,
      detail::result_of_t<Factories()>&...
    >
  >
>
  multi_agent_then_execute_with_shared_inits_returning_default_container(std::false_type, Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  using container_type = typename executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename executor_traits<Executor>::shape_type,
      Future,
      detail::result_of_t<Factories()>&...
    >
  >;

  return executor_traits<Executor>::then_execute(ex, f, container_factory<container_type>{}, shape, fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_default_container()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future, class... Factories,
           class Enable1,
           class Enable2,
           class Enable3,
           class Enable4
          >
typename executor_traits<Executor>::template future<
  typename executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename executor_traits<Executor>::index_type,
      Future,
      detail::result_of_t<Factories()>&...
    >
  >
>
  executor_traits<Executor>
    ::then_execute(typename executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename executor_traits<Executor>::shape_type shape,
                   Future& fut,
                   Factories... shared_factories)
{
  using check_for_member_function = detail::executor_traits_detail::has_multi_agent_then_execute_with_shared_inits_returning_default_container<
    Executor,
    Function,
    Future,
    Factories...
  >;

  return detail::executor_traits_detail::multi_agent_then_execute_with_shared_inits_returning_default_container(check_for_member_function(), ex, f, shape, fut, shared_factories...);
} // end executor_traits::then_execute()


} // end agency

