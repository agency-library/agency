#pragma once

#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/future.hpp>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// strategies for single_agent_when_all_execute_and_select implementation
struct use_single_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_default {};


template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
using select_single_agent_when_all_execute_and_select_implementation = 
  typename std::conditional<
      has_single_agent_when_all_execute_and_select<Executor, Function, TupleOfFutures, Indices...>::value,
      use_single_agent_when_all_execute_and_select_member_function,
      typename std::conditional<
        has_multi_agent_when_all_execute_and_select<Executor, Function, typename new_executor_traits<Executor>::shape_type, TupleOfFutures, Indices...>::value,
        use_multi_agent_when_all_execute_and_select_member_function,
        use_default
      >::type
    >::type;


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(use_single_agent_when_all_execute_and_select_member_function, Executor& ex, Function f, TupleOfFutures&& futures)
{
  return ex.template when_all_execute_and_select<Indices...>(f, std::forward<TupleOfFutures>(futures));
} // end single_agent_when_all_execute_and_select()


template<class Function>
struct single_agent_when_all_execute_and_select_functor
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Args&&... args) const
  {
    f(std::forward<Args>(args)...);
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(use_multi_agent_when_all_execute_and_select_member_function, Executor& ex, Function f, TupleOfFutures&& futures)
{
  // create a multi-agent task with only a single agent
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  using index_type = typename new_executor_traits<Executor>::index_type;

  // XXX should std::move f into the functor
  return ex.template when_all_execute_and_select<Indices...>(single_agent_when_all_execute_and_select_functor<Function>{f}, detail::shape_cast<shape_type>(1), std::forward<TupleOfFutures>(futures));
} // end single_agent_when_all_execute_and_select()


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(use_default, Executor&, Function f, TupleOfFutures&& futures)
{
  // XXX other possible implementations:
  // XXX then_execute(when_all(), select<Indices...>()))
  return agency::when_all_execute_and_select<Indices...>(f, std::forward<TupleOfFutures>(futures));
} // end single_agent_when_all_execute_and_select()


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
                                  TupleOfFutures&& futures)
{
  using implementation_strategy = detail::new_executor_traits_detail::select_single_agent_when_all_execute_and_select_implementation<
    Executor,
    Function,
    typename std::decay<TupleOfFutures>::type,
    Indices...
  >;

  return detail::new_executor_traits_detail::single_agent_when_all_execute_and_select<Indices...>(implementation_strategy(), ex, f, std::forward<TupleOfFutures>(futures));
} // end new_executor_traits::when_all_execute_and_select()


} // end agency


