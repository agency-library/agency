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


template<class Container, class Executor, class Function, class Shape, class Future>
struct has_multi_agent_then_execute_returning_user_specified_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().template then_execute<Container>(
               std::declval<Function>(),
               std::declval<Shape>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<Container,ReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Container, class Executor, class Function, class Shape, class Future>
using has_multi_agent_then_execute_returning_user_specified_container = typename has_multi_agent_then_execute_returning_user_specified_container_impl<Container,Executor,Function,Shape,Future>::type;


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.template then_execute<Container>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_user_specified_container()


template<class Function>
struct multi_agent_then_execute_returning_user_specified_container_functor
{
  mutable Function f;

  template<class Index, class Container, class Arg>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c, Arg& arg) const
  {
    c[idx] = f(idx, arg);
  }

  template<class Index, class Container>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& c) const
  {
    c[idx] = f(idx);
  }
};


template<class Container, class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_then_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  using traits = new_executor_traits<Executor>;

  auto results = traits::template make_ready_future<Container>(ex, shape);

  auto results_and_fut = detail::make_tuple(std::move(results), std::move(fut));

  return traits::template when_all_execute_and_select<0>(ex, multi_agent_then_execute_returning_user_specified_container_functor<Function>{f}, shape, results_and_fut);
} // end multi_agent_then_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
           >
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_user_specified_container<
    Container,
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, shape, fut);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >
>
  multi_agent_then_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::shape_type,
      Future
    >
  >;

  return new_executor_traits<Executor>::template then_execute<container_type>(ex, f, shape, fut);
} // end multi_agent_then_execute_returning_default_container()


template<class Executor, class Future, class Function, class Shape, class ExpectedReturnType>
struct has_multi_agent_then_execute_returning_default_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Shape>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, ExpectedReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Shape, class Future, class ExpectedReturnType>
using has_multi_agent_then_execute_returning_default_container = typename has_multi_agent_then_execute_returning_default_container_impl<Executor,Function,Shape,Future,ExpectedReturnType>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
          >
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::index_type,
      Future
    >
  >
>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    detail::result_of_continuation_t<
      Function,
      typename new_executor_traits<Executor>::index_type,
      Future
    >
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_default_container<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_default_container(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return ex.then_execute(f, shape, fut);
} // end multi_agent_then_execute_returning_void()


template<class Executor, class Function, class Future>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_then_execute_returning_void(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape, Future& fut)
{
  return new_executor_traits<Executor>::when_all_execute_and_select(ex, f, shape, detail::make_tuple(std::move(fut)));
} // end multi_agent_then_execute_returning_void()


template<class Executor, class Function, class Shape, class Future>
struct has_multi_agent_then_execute_returning_void_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Shape>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_void<ReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Shape, class Future>
using has_multi_agent_then_execute_returning_void = typename has_multi_agent_then_execute_returning_void_impl<Executor,Function,Shape,Future>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function, class Future,
           class Enable1,
           class Enable2,
           class Enable3
          >
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::then_execute(typename new_executor_traits<Executor>::executor_type& ex,
                   Function f,
                   typename new_executor_traits<Executor>::shape_type shape,
                   Future& fut)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_then_execute_returning_void<
    Executor,
    Function,
    typename new_executor_traits<Executor>::shape_type,
    Future
  >;

  return detail::new_executor_traits_detail::multi_agent_then_execute_returning_void(check_for_member_function(), ex, f, shape, fut);
} // end new_executor_traits::then_execute()


} // end agency

