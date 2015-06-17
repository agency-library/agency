#pragma once

#include <agency/new_executor_traits.hpp>
#include <agency/future.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_when_all_execute_and_select_implementation_strategies
{

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

// XXX multi-agent execute is recursive on multi_agent_when_all_execute_and_select, so we need to
//     find a better way to recurse but also be able to terminate
//     we want the default to simply call executor_traits::execute() without having to do all this extra checking
//     maybe an executor_traits function should never call when_all_execute_and_select() through executor_traits?
struct use_single_agent_when_all_execute_and_select_with_nested_execute_member_function {};

// XXX there must be a way to generalize what we're doing here
// XXX struct use_single_agent_when_all_execute_and_select_with_nested_async_execute_and_wait

struct use_single_agent_when_all_execute_and_select_with_nested_loop {};


template<class IndexSequence, class Executor, class Function, class Shape, class TupleOfFutures>
using has_multi_agent_when_all_execute_and_select_with_ignored_shared_inits = 
  has_multi_agent_when_all_execute_and_select_with_shared_inits<
    IndexSequence,
    Executor,
    Function,
    Shape,
    TupleOfFutures,
    detail::type_list_repeat<new_executor_traits<Executor>::execution_depth, detail::ignore_t>
  >;


template<class Executor>
struct dummy_functor_taking_index_type
{
  __AGENCY_ANNOTATION
  void operator()(const typename new_executor_traits<Executor>::index_type&) const {}
};


template<class Executor>
using has_multi_agent_execute_returning_void =
  has_multi_agent_execute_returning_void<
    Executor,
    dummy_functor_taking_index_type<Executor>   
  >;


template<class Executor, class Function, class Shape, class TupleOfFutures, size_t... Indices>
using select_multi_agent_when_all_execute_and_select_implemention =
  typename std::conditional<
    has_multi_agent_when_all_execute_and_select<Executor, Function, Shape, TupleOfFutures, Indices...>::value,
    use_multi_agent_when_all_execute_and_select_member_function,
    typename std::conditional<
      has_multi_agent_when_all_execute_and_select_with_ignored_shared_inits<detail::index_sequence<Indices...>, Executor, Function, Shape, TupleOfFutures>::value,
      use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
      typename std::conditional<
        has_multi_agent_execute_returning_void<Executor>::value,
        use_single_agent_when_all_execute_and_select_with_nested_execute_member_function,
        use_single_agent_when_all_execute_and_select_with_nested_loop
      >::type
    >::type
  >::type;

} // end multi_agent_when_all_execute_and_select_implementation_strategies



template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(multi_agent_when_all_execute_and_select_implementation_strategies::use_multi_agent_when_all_execute_and_select_member_function,
                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  return ex.template when_all_execute_and_select<Indices...>(f, shape, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


template<size_t num_ignored_arguments, class Function>
struct multi_agent_when_all_execute_and_select_using_ignored_shared_inits_functor
{
  mutable Function f;

  template<size_t... ArgIndices, class Index, class Tuple>
  __AGENCY_ANNOTATION
  void impl(detail::index_sequence<ArgIndices...>, Index&& idx, Tuple&& arg_tuple) const
  {
    f(idx, std::get<ArgIndices>(std::forward<Tuple>(arg_tuple))...);
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


template<size_t... SelectedIndices, size_t... SharedInitIndices, class Executor, class Function, class TupleOfFutures, class... Types>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<SelectedIndices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(multi_agent_when_all_execute_and_select_implementation_strategies::use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                          detail::index_sequence<SharedInitIndices...>,
                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures, const detail::tuple<Types...>& ignored_shared_inits)
{
  constexpr size_t num_ignored_arguments = sizeof...(Types);
  auto g = multi_agent_when_all_execute_and_select_using_ignored_shared_inits_functor<num_ignored_arguments, Function>{f};

  return ex.template when_all_execute_and_select<SelectedIndices...>(g, shape, std::forward<TupleOfFutures>(futures), std::get<SharedInitIndices>(ignored_shared_inits)...);
} // end multi_agent_when_all_execute_and_select()


template<size_t... SelectedIndices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<SelectedIndices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(multi_agent_when_all_execute_and_select_implementation_strategies::use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function implementation_strategy,
                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  constexpr size_t depth = new_executor_traits<Executor>::execution_depth;

  // create ignored shared initializers
  auto ignored_shared_inits = detail::tuple_repeat<depth>(detail::ignore);

  return multi_agent_when_all_execute_and_select<SelectedIndices...>(implementation_strategy,
                                                                     detail::make_index_sequence<depth>(),
                                                                     ex, f, shape, std::forward<TupleOfFutures>(futures), ignored_shared_inits);
} // end multi_agent_when_all_execute_and_select()


template<class Executor, class Function>
struct multi_agent_when_all_execute_and_select_functor_using_nested_execute
{
  Executor& ex;
  mutable Function f;
  typename new_executor_traits<Executor>::shape_type shape;

  template<class... Args>
  struct inner_functor
  {
    mutable Function f;
    mutable detail::tuple<Args&...> args;

    template<size_t... TupleIndices, class Index>
    __AGENCY_ANNOTATION
    void impl(detail::index_sequence<TupleIndices...>, const Index& idx) const
    {
      f(idx, std::get<TupleIndices>(args)...);
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
    new_executor_traits<Executor>::execute(ex, inner_functor<Args...>{f, detail::tie(args...)}, shape);
  }
};


template<size_t... Indices, class Executor, class Function, class TupleOfFutures>
typename new_executor_traits<Executor>::template future<
  detail::when_all_execute_and_select_result_t<
    detail::index_sequence<Indices...>,
    typename std::decay<TupleOfFutures>::type
  >
>
  multi_agent_when_all_execute_and_select(multi_agent_when_all_execute_and_select_implementation_strategies::use_single_agent_when_all_execute_and_select_with_nested_execute_member_function,
                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  return new_executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, multi_agent_when_all_execute_and_select_functor_using_nested_execute<Executor,Function>{ex,f,shape}, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


template<class Function, class Index, class Shape>
struct multi_agent_when_all_execute_and_select_functor_using_nested_loop
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
  multi_agent_when_all_execute_and_select(multi_agent_when_all_execute_and_select_implementation_strategies::use_single_agent_when_all_execute_and_select_with_nested_loop, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, TupleOfFutures&& futures)
{
  using index_type = typename new_executor_traits<Executor>::index_type;
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  auto g = multi_agent_when_all_execute_and_select_functor_using_nested_loop<Function, index_type, shape_type>{f, shape};

  return new_executor_traits<Executor>::template when_all_execute_and_select<Indices...>(ex, g, std::forward<TupleOfFutures>(futures));
} // end multi_agent_when_all_execute_and_select()


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
  using namespace detail::new_executor_traits_detail::multi_agent_when_all_execute_and_select_implementation_strategies;

  using implementation_strategy = select_multi_agent_when_all_execute_and_select_implemention<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    typename std::decay<TupleOfFutures>::type,
    Indices...
  >;

  return detail::new_executor_traits_detail::multi_agent_when_all_execute_and_select<Indices...>(implementation_strategy(), ex, f, shape, std::forward<TupleOfFutures>(futures));
} // end new_executor_traits::when_all_execute_and_select()


} // end agency


