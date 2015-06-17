#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  multi_agent_async_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.async_execute(f, shape);
} // end multi_agent_async_execute_returning_default_container()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  multi_agent_async_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  return new_executor_traits<Executor>::template async_execute<container_type>(ex, f, shape);
} // end multi_agent_async_returning_default_specified_container()


template<class Executor, class Function, class ExpectedReturnType>
struct has_multi_agent_async_execute_returning_default_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,ExpectedReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class ExpectedReturnType>
using has_multi_agent_async_execute_returning_default_container = typename has_multi_agent_async_execute_returning_default_container_impl<Executor,Function,ExpectedReturnType>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_default_container<
    Executor,
    Function,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_returning_default_container(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::async_execute()


} // end agency

