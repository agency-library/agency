#pragma once

#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace multi_agent_when_all_execute_and_select_implementation_strategies
{

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

struct use_single_agent_when_all_execute_and_select_with_nested_multi_agent_execute {};


template<class IndexSequence, class Executor, class Function, class TupleOfFutures>
using has_multi_agent_when_all_execute_and_select_with_unit_factories = 
  has_multi_agent_when_all_execute_and_select_with_shared_inits<
    IndexSequence,
    Executor,
    Function,
    TupleOfFutures,
    detail::type_list_repeat<executor_traits<Executor>::execution_depth, detail::unit_factory>
  >;


template<class Executor>
struct dummy_functor_taking_index_type
{
  __AGENCY_ANNOTATION
  void operator()(const typename executor_traits<Executor>::index_type&) const {}
};


template<class Executor>
using has_multi_agent_execute_returning_void =
  executor_traits_detail::has_multi_agent_execute_returning_void<
    Executor,
    dummy_functor_taking_index_type<Executor>   
  >;


template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
using select_multi_agent_when_all_execute_and_select_implementation =
  typename std::conditional<
    has_multi_agent_when_all_execute_and_select<Executor, Function, TupleOfFutures, Indices...>::value,
    use_multi_agent_when_all_execute_and_select_member_function,
    typename std::conditional<
      has_multi_agent_when_all_execute_and_select_with_unit_factories<detail::index_sequence<Indices...>, Executor, Function, TupleOfFutures>::value,
      use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
      use_single_agent_when_all_execute_and_select_with_nested_multi_agent_execute
    >::type
  >::type;



template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(use_multi_agent_when_all_execute_and_select_member_function,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  return ex.template when_all_execute_and_select<Indices...>(f, shape, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


template<size_t num_ignored_arguments, class Function>
struct multi_agent_when_all_execute_and_select_ignoring_shared_args_functor
{
  mutable Function f;

  template<size_t... ArgIndices, class Index, class Tuple>
  __AGENCY_ANNOTATION
  void impl(detail::index_sequence<ArgIndices...>, Index&& idx, Tuple&& arg_tuple) const
  {
    agency::detail::invoke(f, idx, std::get<ArgIndices>(std::forward<Tuple>(arg_tuple))...);
  }

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  void operator()(Index&& idx, Args&&... args) const
  {
    // ignore the arguments that come after the futures
    constexpr size_t num_non_ignored_arguments = sizeof...(Args) - num_ignored_arguments;
    impl(detail::make_index_sequence<num_non_ignored_arguments>(), std::forward<Index>(idx), detail::forward_as_tuple(std::forward<Args>(args)...));
  }
};


template<size_t... SelectedIndices, size_t... SharedInitIndices, class Executor, class Function, class TupleOfFutures, class... Factories>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<SelectedIndices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                          detail::index_sequence<SharedInitIndices...>,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures, const detail::tuple<Factories...>& unit_factories)
{
  constexpr size_t num_ignored_arguments = sizeof...(Factories);
  auto g = multi_agent_when_all_execute_and_select_ignoring_shared_args_functor<num_ignored_arguments, Function>{f};

  return ex.template when_all_execute_and_select<SelectedIndices...>(g, shape, std::forward<TupleOfFutures>(futures), std::get<SharedInitIndices>(unit_factories)...);
} // end multi_agent_when_all_execute_and_select()


template<size_t... SelectedIndices, class Executor, class Function, class TupleOfFutures>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<SelectedIndices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function implementation_strategy,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  constexpr size_t depth = executor_traits<Executor>::execution_depth;

  // create unit factories
  auto unit_factory_tuple = detail::tuple_repeat<depth>(detail::unit_factory());

  return multi_agent_when_all_execute_and_select<SelectedIndices...>(implementation_strategy,
                                                                     detail::make_index_sequence<depth>(),
                                                                     ex, f, shape, std::forward<TupleOfFutures>(futures), unit_factory_tuple);
} // end multi_agent_when_all_execute_and_select()


template<class Executor, class Function>
struct multi_agent_when_all_execute_and_select_functor_using_nested_execute
{
  Executor& ex;
  mutable Function f;
  typename executor_traits<Executor>::shape_type shape;

  template<class... Args>
  struct inner_functor
  {
    mutable Function f;
    mutable detail::tuple<Args&...> args;

    template<size_t... TupleIndices, class Index>
    __AGENCY_ANNOTATION
    void impl(detail::index_sequence<TupleIndices...>, const Index& idx) const
    {
      agency::detail::invoke(f, idx, std::get<TupleIndices>(args)...);
    }

    template<class Index>
    __AGENCY_ANNOTATION
    void operator()(const Index& idx) const
    {
      impl(detail::make_index_sequence<sizeof...(Args)>(), idx);
    }
  };

  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(Args&... args) const
  {
    executor_traits<Executor>::execute(ex, inner_functor<Args...>{f, detail::tie(args...)}, shape);
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(use_single_agent_when_all_execute_and_select_with_nested_multi_agent_execute,
                                          Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  return executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, multi_agent_when_all_execute_and_select_functor_using_nested_execute<Executor,Function>{ex,f,shape}, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()



} // end multi_agent_when_all_execute_and_select_implementation_strategies
} // end executor_traits_detail
} // end detail



template<class Executor>
template<size_t... Indices, class Function, class TupleOfFutures>
  typename executor_traits<Executor>::template future<
    detail::when_all_execute_and_select_result_t<
      detail::index_sequence<Indices...>,
      typename std::decay<TupleOfFutures>::type
    >
  >
  executor_traits<Executor>
    ::when_all_execute_and_select(typename executor_traits<Executor>::executor_type& ex,
                                  Function f,
                                  typename executor_traits<Executor>::shape_type shape,
                                  TupleOfFutures&& futures)
{
  namespace ns = detail::executor_traits_detail::multi_agent_when_all_execute_and_select_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_when_all_execute_and_select_implementation<
    Executor,
    Function,
    typename std::decay<TupleOfFutures>::type,
    Indices...
  >;

  return ns::multi_agent_when_all_execute_and_select<Indices...>(implementation_strategy(), ex, f, shape, std::forward<TupleOfFutures>(futures));
} // end executor_traits::when_all_execute_and_select()


} // end agency


