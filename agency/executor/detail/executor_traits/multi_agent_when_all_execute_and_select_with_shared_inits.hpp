#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/shared_parameter_container.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/index_cast.hpp>
#include <agency/detail/invoke.hpp>
#include <type_traits>
#include <utility>
#include <cassert>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace multi_agent_when_all_execute_and_select_with_shared_inits_implementation_strategies
{


struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_single_agent_when_all_execute_and_select_with_nested_terminal_multi_agent_execute_with_shared_inits {};


template<class Function, class Shape>
struct multi_agent_when_all_execute_and_select_with_shared_inits_functor
{
  mutable Function f;
  Shape shape;

  template<size_t depth, class AgentIndex>
  __AGENCY_ANNOTATION
  size_t rank_in_group(const AgentIndex& idx) const
  {
    // to compute the rank of an index at a particular depth,
    // first prepend 0 (1) to idx (shape) to represent an index of the root group (it has none otherwise)
    // XXX seems like index_cast() should just do the right thing for empty indices
    //     it would correspond to a single-agent task
    auto augmented_idx   = detail::tuple_prepend(detail::wrap_scalar(idx), size_t{0});
    auto augmented_shape = detail::tuple_prepend(detail::wrap_scalar(shape), size_t{1});
    
    // take the first depth+1 (plus one because we prepended 1) indices of the index & shape and do an index_cast to size_t
    return detail::index_cast<size_t>(detail::tuple_take<depth+1>(augmented_idx),
                                      detail::tuple_take<depth+1>(augmented_shape),
                                      detail::shape_size(detail::tuple_take<depth+1>(augmented_shape)));
  }

  __agency_exec_check_disable__
  template<size_t... ContainerIndices, class AgentIndex, class TupleOfContainers, class... Types>
  __AGENCY_ANNOTATION
  void impl(detail::index_sequence<ContainerIndices...>, AgentIndex&& agent_idx, TupleOfContainers&& shared_arg_containers, Types&... past_args) const
  {
    agency::detail::invoke(
      f,
      std::forward<AgentIndex>(agent_idx),                                                                // pass the agent index
      past_args...,                                                                                       // pass the arguments coming in from futures
      detail::get<ContainerIndices>(shared_arg_containers)[rank_in_group<ContainerIndices>(agent_idx)]... // pass the arguments coming in from shared parameters
    );
  }

  template<class Index, class TupleOfContainers, class... Types>
  __AGENCY_ANNOTATION
  void operator()(Index&& idx, TupleOfContainers& shared_arg_containers, Types&... past_args) const
  {
    constexpr size_t num_containers = std::tuple_size<TupleOfContainers>::value;
    impl(detail::make_index_sequence<num_containers>(), std::forward<Index>(idx), shared_arg_containers, past_args...);
  }
};


template<class IndexSequence, class Executor, class Function, class TupleOfFutures, class TypeList>
struct select_multi_agent_when_all_execute_and_select_with_shared_inits_implementation_impl;

template<size_t... Indices, class Executor, class Function, class TupleOfFutures, class... Types>
struct select_multi_agent_when_all_execute_and_select_with_shared_inits_implementation_impl<
  detail::index_sequence<Indices...>, Executor, Function, TupleOfFutures, type_list<Types...>
>
{
  using type = typename std::conditional<
    has_multi_agent_when_all_execute_and_select_with_shared_inits<
      detail::index_sequence<Indices...>, Executor, Function, TupleOfFutures, type_list<Types...>
    >::value,
    use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
    typename std::conditional<
      has_multi_agent_when_all_execute_and_select<Executor, Function, TupleOfFutures, Indices...>::value,
      use_multi_agent_when_all_execute_and_select_member_function,
      use_single_agent_when_all_execute_and_select_with_nested_terminal_multi_agent_execute_with_shared_inits
    >::type
  >::type;
};

template<class IndexSequence, class Executor, class Function, class TupleOfFutures, class TypeList>
using select_multi_agent_when_all_execute_and_select_with_shared_inits_implementation = typename select_multi_agent_when_all_execute_and_select_with_shared_inits_implementation_impl<IndexSequence,Executor,Function,TupleOfFutures,TypeList>::type;


template<size_t... Indices, class Executor, class Function, class TupleOfFutures, class... Types>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select_with_shared_inits(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                                            Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures, Types&&... shared_inits)
{
  return ex.template when_all_execute_and_select<Indices...>(f, shape, std::forward<TupleOfFutures>(futures), std::forward<Types>(shared_inits)...);
} // end multi_agent_when_all_execute_and_select_with_shared_inits()


template<size_t... Indices, class Executor, class Function, class TupleOfFutures, class... Factories>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select_with_shared_inits(use_multi_agent_when_all_execute_and_select_member_function,
                                                            Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures, Factories... shared_factories)
{
  // create a tuple of containers holding a shared parameter for each group
  auto shared_param_containers_tuple = executor_traits_detail::make_tuple_of_shared_parameter_containers(ex, shape, shared_factories...);

  // turn it into a future
  auto shared_param_containers_tuple_fut = executor_traits<Executor>::template make_ready_future<decltype(shared_param_containers_tuple)>(ex, std::move(shared_param_containers_tuple));

  // combine the shared parameters with the incoming futures
  // the tuple of containers goes in front of the incoming futures
  auto shared_and_futures = detail::tuple_prepend(std::move(futures), std::move(shared_param_containers_tuple_fut));

  // wrap f with a functor to map container elements to shared parameters
  auto g = multi_agent_when_all_execute_and_select_with_shared_inits_functor<Function, typename executor_traits<Executor>::shape_type>{f, shape};

  // add one to the indices to skip the tuple of containers which was prepended to the tuple of futures
  return ex.template when_all_execute_and_select<(Indices+1)...>(g, shape, std::move(shared_and_futures));
} // end multi_agent_when_all_execute_and_select_with_shared_inits()


// this functor is passed to single-agent when_all_execute_and_select below
// it makes a nested call to terminal multi-agent execute()
template<class Executor, class Function, class... Types>
struct terminal_execute_with_shared_inits_functor
{
  Executor& ex;
  mutable Function f;
  typename executor_traits<Executor>::shape_type shape;
  detail::tuple<Types...> shared_inits;

  // this is the functor we pass to executor_traits::execute() below
  template<class... FutureValueTypes>
  struct nested_functor
  {
    mutable Function f;
    mutable detail::tuple<FutureValueTypes&...> args_from_futures;

    template<size_t... TupleIndices, class Index, class... Args>
    __AGENCY_ANNOTATION
    void impl(detail::index_sequence<TupleIndices...>, const Index& idx, Args&... shared_args) const
    {
      agency::detail::invoke(f, idx, std::get<TupleIndices>(args_from_futures)..., shared_args...);
    }

    template<class Index, class... Args>
    __AGENCY_ANNOTATION
    void operator()(const Index& idx, Args&... shared_args) const
    {
      impl(detail::make_index_sequence<sizeof...(FutureValueTypes)>(), idx, shared_args...);
    }
  };

  template<size_t... TupleIndices, class... Args>
  __AGENCY_ANNOTATION
  void impl(detail::index_sequence<TupleIndices...>, Args&... future_args) const
  {
    //// XXX with polymorphic lambda we'd write something like this:
    //executor_traits<Executor>::execute(ex, [=,&args...](const auto& idx, auto&... shared_args)
    //{
    //  // XXX should use std::invoke()
    //  f(idx, args..., shared_args...);
    //},
    //shape,
    //std::get<Indices>(shared_inits)...
    //);

    auto g = nested_functor<Args...>{f, detail::tie(future_args...)};

    executor_traits<Executor>::execute(ex, g, shape, std::get<TupleIndices>(shared_inits)...);
  }

  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(Args&... future_args) const
  {
    impl(detail::make_index_sequence<sizeof...(Types)>(), future_args...);
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures, class... Types>
typename executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select_with_shared_inits(use_single_agent_when_all_execute_and_select_with_nested_terminal_multi_agent_execute_with_shared_inits,
                                                            Executor& ex, Function f, typename executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures, Types&&... shared_inits)
{
  // XXX with polymorphic lambda we'd write something like this:
  //return executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, [=,&ex](auto&... args)
  //{
  //  executor_traits<Executor>::execute(ex, [=,&args](const auto& idx)
  //  {
  //    f(idx, args...);
  //  },
  //  shape,
  //  shared_inits...);
  //},
  //std::forward<TupleOfFutures>(futures),
  //std::forward<Types>(shared_inits)...
  //);

  using functor_type = terminal_execute_with_shared_inits_functor<Executor,Function,typename std::decay<Types>::type...>;
  functor_type g{ex, f, shape, detail::make_tuple(std::forward<Types>(shared_inits)...)};

  return executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, g, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select_with_shared_inits()


} // end multi_agent_when_all_execute_and_select_with_shared_inits_implementation_strategies
} // end detail
} // end executor_traits_detail


template<class Executor>
template<size_t... Indices, class Function, class TupleOfFutures, class Factory, class... Factories>
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
                                  TupleOfFutures&& futures,
                                  Factory outer_shared_factory,
                                  Factories... inner_shared_factories)
{
  static_assert(executor_traits<Executor>::execution_depth == 1 + sizeof...(Factories), "The number of factories must be equal to the executor's execution_depth.");

  namespace ns = detail::executor_traits_detail::multi_agent_when_all_execute_and_select_with_shared_inits_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_when_all_execute_and_select_with_shared_inits_implementation<
    detail::index_sequence<Indices...>,
    Executor,
    Function,
    typename std::decay<TupleOfFutures>::type,
    detail::type_list<Factory,Factories...>
  >;

  return ns::multi_agent_when_all_execute_and_select_with_shared_inits<Indices...>(implementation_strategy(), ex, f, shape, std::forward<TupleOfFutures>(futures), outer_shared_factory, inner_shared_factories...);
} // end executor_traits::when_all_execute_and_select()


} // end agency

