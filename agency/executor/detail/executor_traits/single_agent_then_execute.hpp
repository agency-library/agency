#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/executor/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/executor/detail/executor_traits/copy_constructible_function.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{
namespace single_agent_then_execute_implementation_strategies
{


// strategies for single_agent_then_execute implementation
struct use_single_agent_then_execute_member_function {};

struct use_single_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_future_traits_then_with_nested_single_agent_execute {};


template<class Executor, class Function, class Future>
using has_multi_agent_when_all_execute_and_select =
  executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    Function,
    detail::tuple<Future>,
    0
  >;


template<class Executor, class Function, class Future>
using select_single_agent_then_execute_implementation = 
  typename std::conditional<
    has_single_agent_then_execute<Executor,Function,Future>::value,
    use_single_agent_then_execute_member_function,
    typename std::conditional<
      has_single_agent_when_all_execute_and_select<Executor,Function,detail::tuple<Future>,0>::value,
      use_single_agent_when_all_execute_and_select_member_function,
      typename std::conditional<
        has_multi_agent_when_all_execute_and_select<Executor,Function,Future>::value,
        use_multi_agent_when_all_execute_and_select_member_function,
        use_future_traits_then_with_nested_single_agent_execute
      >::type
    >::type
  >::type;

} // end single_agent_then_execute_implementation_strategies


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_single_agent_then_execute_member_function,
                            Executor& ex, Function&& f, Future& fut)
{
  return ex.then_execute(std::forward<Function>(f), fut);
} // end single_agent_then_execute()


template<class Function, class Result>
struct single_agent_then_execute_using_single_agent_when_all_execute_and_select_functor
{
  mutable Function f;

  // both f's argument and result are void
  __AGENCY_ANNOTATION
  void operator()() const
  {
    agency::detail::invoke(f);
  }

  // neither f's argument nor result are void
  __agency_exec_check_disable__
  template<class Arg1, class Arg2>
  __AGENCY_ANNOTATION
  void operator()(Arg1& arg1, Arg2& arg2) const
  {
    arg1 = agency::detail::invoke(f, arg2);
  }

  // when the functor receives only a single argument,
  // that means either f's result or parameter is void, but not both
  __agency_exec_check_disable__
  template<class Arg>
  __AGENCY_ANNOTATION
  void operator()(Arg& arg,
                  typename std::enable_if<
                    std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    arg = agency::detail::invoke(f);
  }

  template<class Arg>
  __AGENCY_ANNOTATION
  void operator()(Arg& arg,
                  typename std::enable_if<
                    !std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    agency::detail::invoke(f, arg);
  }
};


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_single_agent_when_all_execute_and_select_member_function,
                            Executor& ex, Function&& f, Future& fut)
{
  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_using_single_agent_when_all_execute_and_select_functor<typename std::decay<Function>::type,result_type>{std::forward<Function>(f)};

  return ex.template when_all_execute_and_select<0>(std::move(g), futures);
} // end single_agent_then_execute()


template<class Function, class Result>
struct single_agent_then_execute_using_multi_agent_when_all_execute_and_select_functor
{
  mutable copy_constructible_function_t<Function> f;

  // both f's argument and result are void
  template<class Index>
  __AGENCY_ANNOTATION
  void operator()(const Index&) const
  {
    agency::detail::invoke(f);
  }

  // neither f's argument nor result are void
  __agency_exec_check_disable__
  template<class Index, class Arg1, class Arg2>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg1& arg1, Arg2& arg2) const
  {
    arg1 = agency::detail::invoke(f,arg2);
  }

  // when the functor receives only a single argument,
  // that means either f's result or parameter is void, but not both
  __agency_exec_check_disable__
  template<class Index, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg& arg,
                  typename std::enable_if<
                    std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    arg = agency::detail::invoke(f);
  }

  template<class Index, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg& arg,
                  typename std::enable_if<
                    !std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    agency::detail::invoke(f, arg);
  }
};


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_multi_agent_when_all_execute_and_select_member_function,
                            Executor& ex, Function f, Future& fut)
{
  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_using_multi_agent_when_all_execute_and_select_functor<typename std::decay<Function>::type,result_type>{std::forward<Function>(f)};

  using shape_type = typename executor_traits<Executor>::shape_type;

  return ex.template when_all_execute_and_select<0>(g, detail::shape_cast<shape_type>(1), futures);
} // end single_agent_then_execute()


template<class Executor, class Function>
struct future_traits_then_with_nested_single_agent_execute_functor
{
  Executor& ex;
  Function f;

  // sizeof...(Args) may only be 0 or 1
  template<class... Args>
  __AGENCY_ANNOTATION
  result_of_t<Function(Args&...)> operator()(Args&... args)
  {
    // XXX maybe should use invoke() inside
    auto g = [&]{ return f(args...); };
    return executor_traits<Executor>::execute(ex, g);
  }
};


// XXX collapse this function and the next one into the same
//     by eliminating the enable_if
__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_future_traits_then_with_nested_single_agent_execute,
                            Executor& ex, Function&& f, Future& fut,
                            typename std::enable_if<
                              !std::is_void<
                                typename future_value<Future>::type
                              >::value
                            >::type* = 0)
{
  // launch f as continuation
  auto continuation = future_traits_then_with_nested_single_agent_execute_functor<Executor,typename std::decay<Function>::type>{ex, std::forward<Function>(f)};
  auto fut2 = future_traits<Future>::then(fut, std::move(continuation));

  // cast to the right type of future
  using value_type2 = typename future_traits<decltype(fut2)>::value_type;
  return executor_traits<Executor>::template future_cast<value_type2>(ex, fut2);
} // end single_agent_then_execute()


__agency_exec_check_disable__
template<class Executor, class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_future_traits_then_with_nested_single_agent_execute,
                            Executor& ex, Function f, Future& fut,
                            typename std::enable_if<
                              std::is_void<
                                typename future_value<Future>::type
                              >::value
                            >::type* = 0)
{
  // launch f as continuation
  auto continuation = future_traits_then_with_nested_single_agent_execute_functor<Executor,typename std::decay<Function>::type>{ex, std::forward<Function>(f)};
  auto fut2 = future_traits<Future>::then(fut, std::move(continuation));

  // cast to the right type of future
  using value_type = typename future_traits<decltype(fut2)>::value_type;
  return executor_traits<Executor>::template future_cast<value_type>(ex, fut2);
} // end single_agent_then_execute()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  executor_traits<Executor>
    ::then_execute(typename executor_traits<Executor>::executor_type& ex,
                   Function&& f,
                   Future& fut)
{
  using namespace detail::executor_traits_detail::single_agent_then_execute_implementation_strategies;

  using implementation_strategy = select_single_agent_then_execute_implementation<Executor,Function,Future>;

  return detail::executor_traits_detail::single_agent_then_execute(implementation_strategy(), ex, std::forward<Function>(f), fut);
} // end executor_traits::then_execute()


} // end agency

