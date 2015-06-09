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


template<size_t... Indices, class Executor, class TupleOfFutures, class Function>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(std::true_type, Executor& ex, TupleOfFutures&& futures, Function f)
{
  return ex.template when_all_execute_and_select<Indices...>(std::forward<TupleOfFutures>(futures), f);
} // end single_agent_when_all_execute_and_select()


template<size_t... Indices, class Executor, class TupleOfFutures, class Function>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(std::false_type, Executor&, TupleOfFutures&& futures, Function f)
{
  // XXX other possible implementations:
  // XXX multi-agent when_all_execute_and_select()
  // XXX then_execute(when_all(), select<Indices...>()))
  return agency::when_all_execute_and_select<Indices...>(std::forward<TupleOfFutures>(futures), f);
} // end single_agent_when_all_execute_and_select()


} // end new_executor_traits_detail
} // end detail



template<class Executor>
template<size_t... Indices, class TupleOfFutures, class Function>
  typename new_executor_traits<Executor>::template future<
    detail::when_all_execute_and_select_result_t<
      detail::index_sequence<Indices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
  new_executor_traits<Executor>
    ::when_all_execute_and_select(typename new_executor_traits<Executor>::executor_type& ex,
                                  TupleOfFutures&& futures,
                                  Function f)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_when_all_execute_and_select<
    Executor,
    typename std::decay<TupleOfFutures>::type,
    Function,
    Indices...
  >;

  return detail::new_executor_traits_detail::single_agent_when_all_execute_and_select<Indices...>(check_for_member_function(), ex, std::forward<TupleOfFutures>(futures), f);
} // end new_executor_traits::when_all_execute_and_select()


} // end agency

