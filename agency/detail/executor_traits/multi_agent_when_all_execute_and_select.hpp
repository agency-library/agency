#pragma once

#include <agency/new_executor_traits.hpp>
#include <agency/future.hpp>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  return ex.template when_all_execute_and_select<Indices...>(f, shape, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


template<class Function, class Index, class Shape>
struct multi_agent_when_all_execute_and_select_functor
{
  mutable Function f;
  Shape shape;

  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(Args&... args) const
  {
    for(Index idx = 0; idx < shape; ++idx)
    {
      f(idx, args...);
    }
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  using index_type = typename new_executor_traits<Executor>::index_type;
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  auto g = multi_agent_when_all_execute_and_select_functor<Function, index_type, shape_type>{f, shape};

  // XXX other possible implementations:
  //     nest an multi-agent execute() inside a single-agent when_all_execute_and_select

  return new_executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, g, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


template<class Executor, class Function, class Shape, class TupleOfFutures, size_t... Indices>
struct has_multi_agent_when_all_execute_and_select_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<Function>(),
               std::declval<Shape>(),
               std::declval<TupleOfFutures>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Shape, class TupleOfFutures, size_t... Indices>
using has_multi_agent_when_all_execute_and_select = typename has_multi_agent_when_all_execute_and_select_impl<Executor, Function, Shape, TupleOfFutures, Indices...>::type;


} // end new_executor_traits_detail
} // end detail



template<class Executor>
template<size_t... Indices, class Function, class TupleOfFutures>
  typename new_executor_traits<Executor>::template future<
    detail::when_all_execute_and_select_result_t<
      detail::index_sequence<Indices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
  new_executor_traits<Executor>
    ::when_all_execute_and_select(typename new_executor_traits<Executor>::executor_type& ex,
                                  Function f,
                                  typename new_executor_traits<Executor>::shape_type shape,
                                  TupleOfFutures&& futures)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    typename std::decay<TupleOfFutures>::type,
    Indices...
  >;

  return detail::new_executor_traits_detail::multi_agent_when_all_execute_and_select<Indices...>(check_for_member_function(), ex, f, shape, std::forward<TupleOfFutures>(futures));
} // end new_executor_traits::when_all_execute_and_select()


} // end agency


