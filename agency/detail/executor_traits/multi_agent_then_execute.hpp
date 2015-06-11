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
  static std::false_type test(...);

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
  static std::false_type test(...);

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
  static std::false_type test(...);

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

