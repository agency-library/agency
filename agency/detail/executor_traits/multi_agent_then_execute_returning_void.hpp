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
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, shape, fut);
} // end multi_agent_then_execute_returning_void()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return new_executor_traits<Executor>::when_all_execute_and_select(ex, f, shape, detail::make_tuple(std::move(fut)));
} // end multi_agent_then_execute_returning_void()


template<class Executor, class Function, class Shape, class Future>
struct has_multi_agent_then_execute_returning_void_impl
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
             std::is_void<ReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Shape, class Future>
using has_multi_agent_then_execute_returning_void = typename has_multi_agent_then_execute_returning_void_impl<Executor,Function,Shape,Future>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
          >
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_void<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_void(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency


