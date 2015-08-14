#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace multi_agent_then_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
{


// 1.
struct use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function {};

using use_strategy_1 = use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function;

template<class Executor, class Container, class Function, class Future, class... Factories>
using has_strategy_1 = has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<
  Container,
  Executor,
  Function,
  Future,
  Factories...
>;


// 2.
struct use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function {};

using use_strategy_2 = use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function;

template<class Executor, class Container, class Function, class Future, class... Factories>
using has_strategy_2 = has_multi_agent_when_all_execute_and_select_with_shared_inits<
  detail::index_sequence<0>,
  Executor,
  test_function_returning_void,
  detail::tuple<
    typename new_executor_traits<Executor>::template future<Container>,
    Future
  >,
  detail::type_list<Factories...>
>;


// 3.
struct use_single_agent_then_execute_with_nested_multi_agent_execute_with_shared_inits {};

using use_strategy_3 = use_single_agent_then_execute_with_nested_multi_agent_execute_with_shared_inits;


template<class Container, class Executor, class Function, class Future, class... Factories>
using select_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_implementation =
  typename std::conditional<
    has_strategy_1<Executor, Container, Function, Future, Factories...>::value,
    use_strategy_1,
    typename std::conditional<
      has_strategy_2<Executor, Container, Function, Future, Factories...>::value,
      use_strategy_2,
      use_strategy_3
    >::type
  >::type;



// strategy 1
template<class Container, class Executor, class Function, class Future, class... Factories>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_member_function,
                                                                                Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut,
                                                                                Factories... shared_factories)
{
  return ex.template then_execute<Container>(f, shape, fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_user_specified_container()



// strategy 2
template<class Function>
struct strategy_2_functor
{
  mutable Function f;

  template<class Index, class Container, class... Args>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c, Args&&... args) const
  {
    c[idx] = f(idx, std::forward<Args>(args)...);
  }
};


template<class Container, class Executor, class Function, class Future, class... Factories>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_with_shared_inits_returning_user_specified_container(use_multi_agent_when_all_execute_and_select_with_shared_inits_member_function,
                                                                                Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  using traits = new_executor_traits<Executor>;

  auto results = traits::template make_ready_future<Container>(ex, shape);

  auto results_and_fut = detail::make_tuple(std::move(results), std::move(fut));

  return ex.template when_all_execute_and_select<0>(strategy_2_functor<Function>{f}, shape, results_and_fut, shared_factories...);
} // end multi_agent_then_execute_with_shared_inits_returning_user_specified_container()


// strategy 3

template<class Container, class Executor, class Function, class T, class Tuple>
struct strategy_3_functor
{
  Executor& ex;
  mutable Function f;
  typename new_executor_traits<Executor>::shape_type shape;
  Tuple shared_inits;

  struct inner_functor
  {
    mutable Function f;
    T& arg2;

    template<class Index, class... Args>
    __AGENCY_ANNOTATION
    typename std::result_of<
      Function(Index, T&, Args...)
    >::type
    operator()(const Index& idx, Args&&... args) const
    {
      return f(idx, arg2, std::forward<Args>(args)...);
    }
  };

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  Container impl(detail::index_sequence<Indices...>, T& val) const
  {
    return new_executor_traits<Executor>::template execute<Container>(ex, inner_functor{f,val}, shape, std::get<Indices>(shared_inits)...);
  }

  __AGENCY_ANNOTATION
  Container operator()(T& val) const
  {
    return impl(detail::make_index_sequence<std::tuple_size<Tuple>::value>(), val);
  }
};


template<class Container, class Executor, class Function, class Tuple>
struct strategy_3_functor<Container,Executor,Function,void,Tuple>
{
  Executor& ex;
  mutable Function f;
  typename new_executor_traits<Executor>::shape_type shape;
  Tuple shared_factories;

  struct inner_functor
  {
    mutable Function f;

    template<class Arg1, class... Args>
    __AGENCY_ANNOTATION
    typename std::result_of<
      Function(Arg1, Args...)
    >::type
    operator()(Arg1&& arg1, Args&&... args) const
    {
      return f(std::forward<Arg1>(arg1), std::forward<Args>(args)...);
    }
  };

  template<size_t... Indices>
  __AGENCY_ANNOTATION
  Container impl(detail::index_sequence<Indices...>) const
  {
    return new_executor_traits<Executor>::template execute<Container>(ex, inner_functor{f}, shape, detail::get<Indices>(shared_factories)...);
  }

  __AGENCY_ANNOTATION
  Container operator()() const
  {
    return impl(detail::make_index_sequence<std::tuple_size<Tuple>::value>());
  }
};


template<class Container, class Executor, class Function, class Future, class... Factories>
__AGENCY_ANNOTATION
strategy_3_functor<
  Container,
  Executor,
  Function,
  typename future_traits<Future>::value_type,
  detail::tuple<
    typename std::decay<Factories>::type...
  >
>
make_strategy_3_functor(Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut, Factories... shared_factories)
{
  auto factory_tuple = detail::make_tuple(shared_factories...);

  return strategy_3_functor<Container,Executor,Function,typename future_traits<Future>::value_type,decltype(factory_tuple)>{ex, f, shape, factory_tuple};
}


template<class Container, class Executor, class Function, class Future, class... Factories>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_with_shared_inits_returning_user_specified_container(use_single_agent_then_execute_with_nested_multi_agent_execute_with_shared_inits,
                                                                                Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut,
                                                                                Factories... shared_factories)
{
  auto g = make_strategy_3_functor<Container>(ex, f, shape, fut, shared_factories...);

  return new_executor_traits<Executor>::then_execute(ex, g, fut);
} // end multi_agent_then_execute_with_shared_inits_returning_user_specified_container()


} // end multi_agent_then_execute_with_shared_inits_returning_user_specified_container_implementation_strategies
} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class Future,
           class... Factories,
           class Enable1,
           class Enable2,
           class Enable3,
           class Enable4
           >
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut,
                   Factories... shared_factories)
{
  namespace ns = detail::new_executor_traits_detail::multi_agent_then_execute_with_shared_inits_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_implementation<
    Container,
    Executor,
    Function,
    Future,
    Factories...
  >;

  return ns::multi_agent_then_execute_with_shared_inits_returning_user_specified_container<Container>(implementation_strategy(), ex, f, shape, fut, shared_factories...);
} // end new_executor_traits::then_execute()


} // end agency

