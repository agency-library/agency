#pragma once

#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/future.hpp>
#include <agency/detail/select.hpp>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace single_agent_when_all_execute_and_select_implementation_strategies
{


// strategies for single_agent_when_all_execute_and_select implementation
struct use_single_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_when_all_and_single_agent_then_execute {};


template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
using select_single_agent_when_all_execute_and_select_implementation = 
  typename std::conditional<
      has_single_agent_when_all_execute_and_select<Executor, Function, TupleOfFutures, Indices...>::value,
      use_single_agent_when_all_execute_and_select_member_function,
      typename std::conditional<
        has_multi_agent_when_all_execute_and_select<Executor, Function, typename new_executor_traits<Executor>::shape_type, TupleOfFutures, Indices...>::value,
        use_multi_agent_when_all_execute_and_select_member_function,
        use_when_all_and_single_agent_then_execute
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


template<class IndexSequence, class TupleOfFutures>
struct when_all_from_tuple_result_impl;

template<size_t... Indices, class TupleOfFutures>
struct when_all_from_tuple_result_impl<index_sequence<Indices...>, TupleOfFutures>
  : when_all_result<
      typename std::tuple_element<
        Indices,
        TupleOfFutures
      >::type...
    >
{};

template<class TupleOfFutures>
struct when_all_from_tuple_result : when_all_from_tuple_result_impl<
  make_index_sequence<std::tuple_size<TupleOfFutures>::value>,
  TupleOfFutures
>
{};

template<class TupleOfFutures>
using when_all_from_tuple_result_t = typename when_all_from_tuple_result<TupleOfFutures>::type;


template<size_t... Indices, class Executor, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  when_all_from_tuple_result_t<
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_from_tuple_impl(index_sequence<Indices...>, Executor& ex, TupleOfFutures&& futures)
{
  return new_executor_traits<Executor>::when_all(ex, std::get<Indices>(std::forward<TupleOfFutures>(futures))...);
}


template<class Executor, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  when_all_from_tuple_result_t<
    typename std::decay<TupleOfFutures>::type
  >
>
  when_all_from_tuple(Executor& ex, TupleOfFutures&& futures)
{
  constexpr size_t num_futures = std::tuple_size<
    typename std::decay<TupleOfFutures>::type
  >::value;

  return when_all_from_tuple_impl(detail::make_index_sequence<num_futures>(), ex, std::forward<TupleOfFutures>(futures));
}


template<size_t num_nonvoid_arguments, class Function, class IndexSequence>
struct invoke_and_select;


template<size_t num_nonvoid_arguments, class Function, size_t... SelectedIndices>
struct invoke_and_select<num_nonvoid_arguments, Function, index_sequence<SelectedIndices...>>
{
  mutable Function f;

  template<size_t... Indices, class Tuple>
  __AGENCY_ANNOTATION
  void invoke_f_and_ignore_result_impl(index_sequence<Indices...>, Tuple& args) const
  {
    f(detail::get<Indices>(args)...);
  }

  template<class Tuple>
  __AGENCY_ANNOTATION
  void invoke_f_and_ignore_result(Tuple& args) const
  {
    invoke_f_and_ignore_result_impl(detail::make_tuple_indices(args), args);
  }

  // 0-element Tuples unwrap to void
  template<class Tuple,
           class = typename std::enable_if<
             (std::tuple_size<
                typename std::decay<Tuple>::type
              >::value == 0)
           >::type>
  __AGENCY_ANNOTATION
  static void unwrap_and_move(Tuple&&) {}

  // 1-element Tuples unwrap and move to their first element
  template<class Tuple,
           class = typename std::enable_if<
             (std::tuple_size<
                typename std::decay<Tuple>::type
              >::value == 1)
           >::type>
  __AGENCY_ANNOTATION
  static typename std::decay<
    typename std::tuple_element<
      0,
      typename std::decay<Tuple>::type
    >::type
  >::type
    unwrap_and_move(Tuple&& t)
  {
    return std::get<0>(std::move(t));
  }

  // n-element Tuples unwrap and move to a tuple of values 
  template<class Tuple,
           class = typename std::enable_if<
             (std::tuple_size<
                typename std::decay<Tuple>::type
              >::value > 1)
           >::type>
  __AGENCY_ANNOTATION
  static decay_tuple_t<Tuple>
    unwrap_and_move(Tuple&& t)
  {
    return decay_tuple_t<Tuple>(std::move(t));
  }


  // usually, we receive all the arguments to f in a tuple
  template<class Tuple>
  __AGENCY_ANNOTATION
  auto operator()(Tuple& args) const
    -> decltype(
         unwrap_and_move(
           detail::select_from_tuple<SelectedIndices...>(args)
         )
       )
  {
    invoke_f_and_ignore_result(args);

    // get the selection from the tuple of arguments
    auto selection = detail::select_from_tuple<SelectedIndices...>(std::move(args));

    // unwrap the selection
    return unwrap_and_move(std::move(selection));
  }
};


template<class Function, size_t... SelectedIndices>
struct invoke_and_select<0, Function, index_sequence<SelectedIndices...>>
{
  mutable Function f;

  // when there are no arguments to f, there is nothing to receive
  __AGENCY_ANNOTATION
  void operator()() const
  {
    f();

    return detail::select<SelectedIndices...>();
  }
};


template<class Function, size_t... SelectedIndices>
struct invoke_and_select<1, Function, index_sequence<SelectedIndices...>>
{
  mutable Function f;

  // when there is only one argument for f, we don't receive a tuple
  // since select() returns a reference, we decay it to a value when we return
  template<class Arg>
  __AGENCY_ANNOTATION
  decay_if_not_void_t<
    select_result_t<index_sequence<SelectedIndices...>, Arg&&>
  >
    operator()(Arg& arg) const
  {
    // invoke f
    f(arg);

    // return a selection from the argument (either ignore it, or move it along)
    return detail::select<SelectedIndices...>(std::move(arg));
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  single_agent_when_all_execute_and_select(use_when_all_and_single_agent_then_execute, Executor& ex, Function f, TupleOfFutures&& futures)
{
  // join the futures into a single one
  auto fut = when_all_from_tuple(ex, std::forward<TupleOfFutures>(futures));

  // count the number of non-void arguments to f
  using future_types = tuple_elements<typename std::decay<TupleOfFutures>::type>;
  using value_types  = type_list_map<future_value, future_types>;
  using nonvoid_value_types = type_list_filter<is_not_void, value_types>;
  constexpr size_t num_nonvoid_arguments = type_list_size<nonvoid_value_types>::value;

  // map Indices... to corresponding indices of the filtered argument list
  // pass those remapped indices to invoke_and_select
  using flag_sequence = type_list_index_map<is_not_void, value_types>;
  using scanned_flags = exclusive_scan_index_sequence<0, flag_sequence>;
  using mapped_indices = index_sequence_gather<index_sequence<Indices...>, scanned_flags>;

  // create a functor which will pass the arguments to f and return a selection of those arguments
  auto g = invoke_and_select<num_nonvoid_arguments, Function, mapped_indices>{f};

  using result_type = detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >;

  return new_executor_traits<Executor>::then_execute(ex, g, fut);
} // end single_agent_when_all_execute_and_select()


} // end single_agent_when_all_execute_and_select_implementation_strategies
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
  namespace ns = detail::new_executor_traits_detail::single_agent_when_all_execute_and_select_implementation_strategies;

  using implementation_strategy = ns::select_single_agent_when_all_execute_and_select_implementation<
    Executor,
    Function,
    typename std::decay<TupleOfFutures>::type,
    Indices...
  >;

  return ns::single_agent_when_all_execute_and_select<Indices...>(implementation_strategy(), ex, f, std::forward<TupleOfFutures>(futures));
} // end new_executor_traits::when_all_execute_and_select()


} // end agency


