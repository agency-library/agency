#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type, typename std::result_of<Factories()>::type&...)
    >::type
  >
>
  multi_agent_async_execute_with_shared_inits_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  return ex.async_execute(f, shape, shared_factories...);
} // end multi_agent_async_execute_with_shared_inits_returning_default_container()


template<class Executor, class Function, class... Factories>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type, typename std::result_of<Factories()>::type&...)
    >::type
  >
>
  multi_agent_async_execute_with_shared_inits_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Factories... shared_factories)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type, typename std::result_of<Factories()>::type&...)
    >::type
  >;

  return new_executor_traits<Executor>::async_execute(ex, f, container_factory<container_type>{}, shape, shared_factories...);
} // end multi_agent_async_with_shared_inits_returning_default_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class... Factories,
           class Enable1,
           class Enable2>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type, typename std::result_of<Factories()>::type&...)
    >::type
  >
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape,
                    Factories... shared_factories)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_with_shared_inits_returning_default_container<
    Executor,
    Function,
    Factories...
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_with_shared_inits_returning_default_container(check_for_member_function(), ex, f, shape, shared_factories...);
} // end new_executor_traits::async_execute()


} // end agency

