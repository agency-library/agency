#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace when_all_implementation_strategies
{


struct swallow
{
  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(Args&&...) const {}
};


// if .when_all() exists, just use it
struct use_when_all_member_function {};

// if either single-agent or multi-agent .when_all_execute_and_select() exists,
// call single-agent when_all_execute_and_select() via executor_traits
// the appropriate member function will get called
struct use_single_agent_when_all_execute_and_select {};

// otherwise, call single-agent then_execute() via executor_traits
struct use_single_agent_then_execute {};


template<class IndexSequence, class Executor, class... Futures>
struct has_single_agent_when_all_execute_and_select_impl;

template<size_t... Indices, class Executor, class... Futures>
struct has_single_agent_when_all_execute_and_select_impl<index_sequence<Indices...>, Executor, Futures...>
{
  using type = new_executor_traits_detail::has_single_agent_when_all_execute_and_select<
    Executor,
    swallow,
    detail::tuple<Futures...>,
    Indices...
  >;
};


template<class Executor, class... Futures>
using has_single_agent_when_all_execute_and_select = typename has_single_agent_when_all_execute_and_select_impl<
  // XXX workaround nvbug 1668651
  //detail::index_sequence_for<Futures...>, Executor, Futures...
  // XXX workaround nvbug 1668718
  //detail::make_index_sequence<sizeof...(Futures)>, Executor, Futures...
  detail::make_index_sequence<sizeof_parameter_pack<Futures...>::value>, Executor, Futures...
>::type;


template<class IndexSequence, class Executor, class... Futures>
struct has_multi_agent_when_all_execute_and_select_impl;

template<size_t... Indices, class Executor, class... Futures>
struct has_multi_agent_when_all_execute_and_select_impl<index_sequence<Indices...>, Executor, Futures...>
{
  using type = new_executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    swallow,
    detail::tuple<Futures...>,
    Indices...
  >;
};


template<class Executor, class... Futures>
using has_multi_agent_when_all_execute_and_select = typename has_multi_agent_when_all_execute_and_select_impl<
  // XXX workaround nvbug 1668651
  //detail::index_sequence_for<Futures...>, Executor, Futures...
  // XXX workaround nvbug 1668718
  //detail::make_index_sequence<sizeof...(Futures)>, Executor, Futures...
  detail::make_index_sequence<sizeof_parameter_pack<Futures...>::value>, Executor, Futures...
>::type;


template<class Executor, class... Futures>
using select_when_all_implementation =
  typename std::conditional<
    has_when_all<Executor,Futures...>::value,
    use_when_all_member_function,
    typename std::conditional<
      (has_single_agent_when_all_execute_and_select<Executor,Futures...>::value || has_multi_agent_when_all_execute_and_select<Executor,Futures...>::value),
      use_single_agent_when_all_execute_and_select,
      use_single_agent_then_execute
    >::type
  >::type;


} // end when_all_implementation_strategies


__agency_hd_warning_disable__
template<class Executor, class... Futures>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  detail::when_all_result_t<
    typename std::decay<Futures>::type...
  >
>
  when_all(when_all_implementation_strategies::use_when_all_member_function,
           Executor& ex, Futures&&... futures)
{
  return ex.when_all(std::forward<Futures>(futures)...);
} // end when_all()


template<size_t... Indices, class Executor, class... Futures>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  detail::when_all_result_t<
    typename std::decay<Futures>::type...
  >
>
  when_all_single_agent_when_all_execute_and_select_impl(detail::index_sequence<Indices...>, Executor& ex, Futures&&... futures)
{
  auto tuple_of_futures = std::make_tuple(std::move(futures)...);
  return new_executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, when_all_implementation_strategies::swallow(), std::move(tuple_of_futures));
} // end when_all()


template<class Executor, class... Futures>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  detail::when_all_result_t<
    typename std::decay<Futures>::type...
  >
>
  when_all(when_all_implementation_strategies::use_single_agent_when_all_execute_and_select implementation_strategy,
           Executor& ex, Futures&&... futures)
{
  return new_executor_traits_detail::when_all_single_agent_when_all_execute_and_select_impl(detail::index_sequence_for<Futures...>(), ex, std::forward<Futures>(futures)...);
} // end when_all()


template<class IndexSequence, class Executor, class Future, class... Futures>
struct when_all_functor;


template<size_t... Indices, class Executor, class HeadFuture, class... TailFutures>
struct when_all_functor<index_sequence<Indices...>, Executor, HeadFuture, TailFutures...>
{
  Executor& exec;
  mutable tuple<TailFutures...> tail_futures;

  // this functor is so complicated because it needs to return:
  // void, when HeadFuture & TailFutures all have void value_type
  // a single T when there is only a single non-void futures in HeadFuture & TailFutures
  // a tuple when there is more than one non-void future in HeadFuture & TailFutures

  using tail_value_types     = type_list<future_value_t<TailFutures>...>;

  template<class T>
  using is_non_void = std::integral_constant<
    bool,
    !std::is_void<T>::value
  >;

  using non_void_tail_value_types = type_list_filter<
    is_non_void,
    tail_value_types
  >;

  using num_tail_values = type_list_size<non_void_tail_value_types>;

  // this function returns the result of the collection of tail futures
  // it can be void, a single T, or a tuple with size > 1
  __AGENCY_ANNOTATION
  detail::when_all_result_t<TailFutures...>
    get_tail() const
  {
    return new_executor_traits<Executor>::when_all(exec, std::get<Indices>(tail_futures)...).get();
  }

  // get_tail_as_tuple() wraps result of get_tail() to ensure a tuple is returned

  // for n == 0, create an empty tuple
  __AGENCY_ANNOTATION
  detail::tuple<>
    get_tail_as_tuple(std::integral_constant<size_t, 0>) const
  {
    get_tail();
    return detail::make_tuple();
  }

  // for n == 1, create a single element tuple
  __AGENCY_ANNOTATION
  detail::tuple<when_all_result_t<TailFutures...>>
    get_tail_as_tuple(std::integral_constant<size_t, 1>) const
  {
    return detail::make_tuple(get_tail());
  }

  // for n > 1, the tail is already a tuple
  template<size_t n>
  __AGENCY_ANNOTATION
  detail::when_all_result_t<TailFutures...>
    get_tail_as_tuple(std::integral_constant<size_t, n>) const
  {
    return get_tail();
  }

  template<class Arg>
  __AGENCY_ANNOTATION
  detail::when_all_result_t<HeadFuture,TailFutures...>
    operator()(Arg& arg) const
  {
    // XXX a more efficient version of this function might recurse by splitting the list of futures
    //     into two and calling when_all() twice instead of once

    // wrap up the head into a tuple
    auto head_as_tuple = detail::make_tuple(std::move(arg));
    
    // get the tail of values as a tuple
    auto tail_as_tuple = get_tail_as_tuple(num_tail_values());

    // concatenate the head and tail together
    auto full_tuple_of_values = detail::tuple_cat(std::move(head_as_tuple), std::move(tail_as_tuple));

    // if the result is only a single element, we need to unwrap the tuple
    return detail::unwrap_single_element_tuple(std::move(full_tuple_of_values));
  }

  __AGENCY_ANNOTATION
  detail::when_all_result_t<HeadFuture, TailFutures...> operator()() const
  {
    // XXX a more efficient version of this function might recurse by splitting the list of futures
    //     into two and calling when_all() twice instead of once
    
    // nothing to prepend, just return the tail
    return get_tail();
  }
};


template<class Executor, class Future, class... Futures>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  detail::when_all_result_t<
    typename std::decay<Future>::type,
    typename std::decay<Futures>::type...
  >
>
  when_all(when_all_implementation_strategies::use_single_agent_then_execute,
           Executor& ex, Future&& future, Futures&&... futures)
{
  auto functor = when_all_functor<detail::index_sequence_for<Futures...>, Executor, typename std::decay<Future>::type, typename std::decay<Futures>::type...>{ex, detail::make_tuple(std::move(futures)...)};

  return new_executor_traits<Executor>::then_execute(ex, std::move(functor), future);
} // end when_all()


template<class Executor>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<void>
  when_all(when_all_implementation_strategies::use_single_agent_then_execute,
           Executor& ex)
{
  // no futures to join, return an immediately ready future
  return new_executor_traits<Executor>::template make_ready_future<void>(ex);
} // end when_all()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class... Futures>
__AGENCY_ANNOTATION
typename new_executor_traits<Executor>::template future<
  detail::when_all_result_t<
    typename std::decay<Futures>::type...
  >
>
  new_executor_traits<Executor>
    ::when_all(typename new_executor_traits<Executor>::executor_type& ex, Futures&&... futures)
{
  using namespace detail::new_executor_traits_detail::when_all_implementation_strategies;

  using implementation_strategy = select_when_all_implementation<
    Executor,
    typename std::decay<Futures>::type...
  >;

  return detail::new_executor_traits_detail::when_all(implementation_strategy(), ex, std::forward<Futures>(futures)...);
} // end new_executor_traits::when_all()


} // end agency

