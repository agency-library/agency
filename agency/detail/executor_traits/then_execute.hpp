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


template<class Executor, class Future, class Function>
struct has_single_agent_then_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().then_execute(
               *std::declval<Future*>(),
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Future, class Function>
using has_single_agent_then_execute = typename has_single_agent_then_execute_impl<Executor,Future,Function>::type;


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(std::true_type, Executor& ex, Future& fut, Function f)
{
  return ex.then_execute(ex, fut, f);
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


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  single_agent_then_execute(std::false_type, Executor& ex, Future& fut, Function f)
{
  // XXX other possible implementations:
  // XXX multi-agent then_execute()
  // XXX future_traits<Future>::then(f);

  using arg_type = typename future_traits<Future>::value_type;
  using result_type = detail::result_of_continuation_t<Function,Future>;

  auto result_future = new_executor_traits<Executor>::template make_ready_future<result_type>(ex);

  auto futures = detail::make_tuple(std::move(result_future), std::move(fut));

  auto g = single_agent_then_execute_functor<Function,result_type>{f};

  return new_executor_traits<Executor>::template when_all_execute_and_select<0>(ex, futures, g);
} // end single_agent_then_execute()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Future, class Function>
typename new_executor_traits<Executor>::template future<
  detail::result_of_continuation_t<Function,Future>
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Future& fut,
                   Function f)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_then_execute<
    Executor,
    Future,
    Function
  >;

  return detail::new_executor_traits_detail::single_agent_then_execute(check_for_member_function(), ex, fut, f);
} // end new_executor_traits::then_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Container, class Executor, class Future, class Function, class Shape>
struct has_multi_agent_then_execute_returning_user_specified_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().template then_execute<Container>(
               *std::declval<Future*>(),
               std::declval<Function>(),
               std::declval<Shape>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<Container,ReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Container, class Executor, class Future, class Function, class Shape>
using has_multi_agent_then_execute_returning_user_specified_container = typename has_multi_agent_then_execute_returning_user_specified_container_impl<Container,Executor,Future,Function,Shape>::type;


template<class Container, class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::true_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template then_execute<Container>(ex, fut, f, shape);
} // end multi_agent_then_execute_returning_user_specified_container()


template<class Function>
struct multi_agent_then_execute_returning_user_specified_container_functor
{
  mutable Function f;

  template<class Container, class Arg, class Index>
  __AGENCY_ANNOTATION
  void operator()(Container& c, Arg& arg, const Index& idx) const
  {
    c[idx] = f(arg, idx);
  }

  template<class Container, class Index>
  __AGENCY_ANNOTATION
  void operator()(Container& c, const Index& idx) const
  {
    c[idx] = f(idx);
  }
};


template<class Container, class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::false_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using traits = new_executor_traits<Executor>;

  auto results = traits::template make_ready_future<Container>(ex, shape);

  auto results_and_fut = detail::make_tuple(std::move(results), std::move(fut));

  return traits::template when_all_execute_and_select<0>(ex, results_and_fut, multi_agent_then_execute_returning_user_specified_container_functor<Function>{f}, shape);
} // end multi_agent_then_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Future, class Function>
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Future& fut,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_user_specified_container<
    Container,
    Executor,
    Future,
    Function,
    typename new_executor_traits<Executor>::shape_type
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, fut, f, shape);
} // end new_executor_traits::then_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      Future,
      typename new_executor_traits<Executor>::shape_type
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::true_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.then_execute(fut, f, shape);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      Future,
      typename new_executor_traits<Executor>::shape_type
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::false_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      Future,
      typename new_executor_traits<Executor>::shape_type
    >
  >;

  return new_executor_traits<Executor>::template then_execute<container_type>(ex, fut, f, shape);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Future, class Function, class Shape, class ExpectedReturnType>
struct has_multi_agent_then_execute_returning_default_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               *std::declval<Future*>(),
               std::declval<Function>(),
               std::declval<Shape>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, ExpectedReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Future, class Function, class Shape, class ExpectedReturnType>
using has_multi_agent_then_execute_returning_default_container = typename has_multi_agent_then_execute_returning_default_container_impl<Executor,Future,Function,Shape,ExpectedReturnType>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Future, class Function,
           class Enable1,
           class Enable2
          >
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      Future,
      typename new_executor_traits<Executor>::index_type
    >
  >
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Future& fut,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      Future,
      typename new_executor_traits<Executor>::index_type
    >
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_default_container<
    Executor,
    Future,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_default_container(check_for_member_function(), ex, fut, f, shape);
} // end new_executor_traits::then_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::true_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.then_execute(fut, f, shape);
} // end multi_agent_then_execute_returning_void()


template<class Executor, class Future, class Function>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::false_type, Executor& ex, Future& fut, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return new_executor_traits<Executor>::when_all_execute_and_select(ex, detail::make_tuple(std::move(fut)), f, shape);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Future, class Function, class Shape>
struct has_multi_agent_then_execute_returning_void_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               *std::declval<Future*>(),
               std::declval<Function>(),
               std::declval<Shape>()
             )
           ),
           class = typename std::enable_if<
             std::is_void<ReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Future, class Function, class Shape>
using has_multi_agent_then_execute_returning_void = typename has_multi_agent_then_execute_returning_void_impl<Executor,Future,Function,Shape>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Future, class Function,
           class Enable1,
           class Enable2
          >
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Future& fut,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_void<
    Executor,
    Future,
    Function,
    typename new_executor_traits<Executor>::shape_type
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_void(check_for_member_function(), ex, fut, f, shape);
} // end new_executor_traits::then_execute()


} // end agency

