#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class Future>
struct has_single_agent_then_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               *std::declval<Future*>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Future>
using has_single_agent_then_execute = typename has_single_agent_then_execute_impl<Executor,Function,Future>::type;


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(std::true_type, Executor& ex, Function f, Future& fut)
{
  return ex.then_execute(ex, f, fut);
} // end single_agent_then_execute()


template<class Function, class Result>
struct single_agent_then_execute_functor
{
  mutable Function f;

  // both f's argument and result are void
  __AGENCY_ANNOTATION
  void operator()() const
  {
    f();
  }

  // neither f's argument nor result are void
  template<class Arg1, class Arg2>
  __AGENCY_ANNOTATION
  void operator()(Arg1& arg1, Arg2& arg2) const
  {
    arg1 = f(arg2);
  }

  // when the functor receives only a single argument,
  // that means either f's result or parameter is void, but not both
  template<class Arg>
  __AGENCY_ANNOTATION
  void operator()(Arg& arg,
                  typename std::enable_if<
                    std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    arg = f();
  }

  template<class Arg>
  __AGENCY_ANNOTATION
  void operator()(Arg& arg,
                  typename std::enable_if<
                    !std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    f(arg);
  }
};


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(std::false_type, Executor& ex, Function f, Future& fut)
{
  // XXX other possible implementations:
  // XXX multi-agent then_execute()
  // XXX future_traits<Future>::then(f);

  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = new_executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_functor<Function,result_type>{f};

  return new_executor_traits<Executor>::template when_all_execute_and_select<0>(ex, g, futures);
} // end single_agent_then_execute()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   Future& fut)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_then_execute<
    Executor,
    Function,
    Future
  >;

  return detail::new_executor_traits_detail::single_agent_then_execute(check_for_member_function(), ex, f, fut);
} // end new_executor_traits::then_execute()


} // end agency

