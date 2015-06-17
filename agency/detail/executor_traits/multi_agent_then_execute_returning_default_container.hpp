#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, shape, fut);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >;

  return new_executor_traits<Executor>::template then_execute<container_type>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Future, class Function, class Shape, class ExpectedReturnType>
struct has_multi_agent_then_execute_returning_default_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Shape>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, ExpectedReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Shape, class Future, class ExpectedReturnType>
using has_multi_agent_then_execute_returning_default_container = typename has_multi_agent_then_execute_returning_default_container_impl<Executor,Function,Shape,Future,ExpectedReturnType>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
          >
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::index_type,
      Future
    >
  >
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::index_type,
      Future
    >
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_default_container<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_default_container(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency

