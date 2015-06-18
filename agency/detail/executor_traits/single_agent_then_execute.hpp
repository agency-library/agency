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
namespace single_agent_then_execute_implementation_strategies
{


// strategies for single_agent_then_execute implementation
struct use_single_agent_then_execute_member_function {};

struct use_single_agent_when_all_execute_and_select_member_function {};

struct use_multi_agent_when_all_execute_and_select_member_function {};

struct use_future_traits_then {};


template<class Executor, class Function, class Future>
using has_multi_agent_when_all_execute_and_select =
  new_executor_traits_detail::has_multi_agent_when_all_execute_and_select<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
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
        use_future_traits_then
      >::type
    >::type
  >::type;

} // end single_agent_then_execute_implementation_strategies


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_single_agent_then_execute_member_function,
                            Executor& ex, Function f, Future& fut)
{
  return ex.then_execute(f, fut);
} // end single_agent_then_execute()


template<class Function, class Result>
struct single_agent_then_execute_using_single_agent_when_all_execute_and_select_functor
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
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_single_agent_when_all_execute_and_select_member_function,
                            Executor& ex, Function f, Future& fut)
{
  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = new_executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_using_single_agent_when_all_execute_and_select_functor<Function,result_type>{f};

  return ex.template when_all_execute_and_select<0>(g, futures);
} // end single_agent_then_execute()


template<class Function, class Result>
struct single_agent_then_execute_using_multi_agent_when_all_execute_and_select_functor
{
  mutable Function f;

  // both f's argument and result are void
  template<class Index>
  __AGENCY_ANNOTATION
  void operator()(const Index&) const
  {
    f();
  }

  // neither f's argument nor result are void
  template<class Index, class Arg1, class Arg2>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg1& arg1, Arg2& arg2) const
  {
    arg1 = f(arg2);
  }

  // when the functor receives only a single argument,
  // that means either f's result or parameter is void, but not both
  template<class Index, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg& arg,
                  typename std::enable_if<
                    std::is_same<Result,Arg>::value
                  >::type* = 0) const
  {
    arg = f();
  }

  template<class Index, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Arg& arg,
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
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_multi_agent_when_all_execute_and_select_member_function,
                            Executor& ex, Function f, Future& fut)
{
  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = new_executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_using_multi_agent_when_all_execute_and_select_functor<Function,result_type>{f};

  using shape_type = typename new_executor_traits<Executor>::shape_type;

  return ex.template when_all_execute_and_select<0>(g, detail::shape_cast<shape_type>(1), futures);
} // end single_agent_then_execute()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_future_traits_then,
                            Executor&, Function f, Future& fut,
                            typename std::enable_if<
                              !std::is_void<
                                typename future_value<Future>::type
                              >::value
                            >::type* = 0)
{
  // XXX should actually use future_traits here
  return agency::detail::then(fut, [=](Future& fut)
  {
    auto arg = fut.get();
    return f(arg);
  });
} // end single_agent_then_execute()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(single_agent_then_execute_implementation_strategies::use_future_traits_then,
                            Executor&, Function f, Future& fut,
                            typename std::enable_if<
                              std::is_void<
                                typename future_value<Future>::type
                              >::value
                            >::type* = 0)
{
  // XXX should actually use future_traits here
  return agency::detail::then(fut, [=](Future& fut)
  {
    return f();
  });
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
  using namespace detail::new_executor_traits_detail::single_agent_then_execute_implementation_strategies;

  using implementation_strategy = select_single_agent_then_execute_implementation<Executor,Function,Future>;

  return detail::new_executor_traits_detail::single_agent_then_execute(implementation_strategy(), ex, f, fut);
} // end new_executor_traits::then_execute()


} // end agency

