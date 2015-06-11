#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::true_type, Executor& ex, Function f)
{
  return ex.async_execute(f);
} // end single_agent_async_execute()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  single_agent_async_execute(std::false_type, Executor& ex, Function f)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::then_execute(ex, ready, f);
} // end single_agent_async_execute()


template<class Executor, class Function>
struct has_single_agent_async_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function>
using has_single_agent_async_execute = typename has_single_agent_async_execute_impl<Executor,Function>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function>
typename new_executor_traits<Executor>::template future<
  typename std::result_of<Function()>::type
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_async_execute<
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::single_agent_async_execute(check_for_member_function(), ex, f);
} // end new_executor_traits::async_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Container, class Executor, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_returning_user_specified_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template async_execute<Container>(f, shape);
} // end multi_agent_async_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
typename new_executor_traits<Executor>::template future<Container>
  multi_agent_async_execute_returning_user_specified_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::template then_execute<Container>(ex, ready, f, shape);
} // end multi_agent_async_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
struct has_multi_agent_async_execute_returning_user_specified_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().template async_execute<Container>(
               std::declval<Function>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,Container>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Container, class Executor, class Function>
using has_multi_agent_async_execute_returning_user_specified_container = typename has_multi_agent_async_execute_returning_user_specified_container_impl<Container,Executor,Function>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Container, class Function>
typename new_executor_traits<Executor>::template future<Container>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_user_specified_container<
    Container,
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_returning_user_specified_container<Container>(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::async_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  multi_agent_async_execute_returning_default_container(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.async_execute(f, shape);
} // end multi_agent_async_execute_returning_default_container()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  multi_agent_async_execute_returning_default_container(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  using container_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  return new_executor_traits<Executor>::template async_execute<container_type>(ex, f, shape);
} // end multi_agent_async_returning_default_specified_container()


template<class Executor, class Function, class ExpectedReturnType>
struct has_multi_agent_async_execute_returning_default_container_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,ExpectedReturnType>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class ExpectedReturnType>
using has_multi_agent_async_execute_returning_default_container = typename has_multi_agent_async_execute_returning_default_container_impl<Executor,Function,ExpectedReturnType>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename new_executor_traits<Executor>::template future<
  typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >
>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using expected_return_type = typename new_executor_traits<Executor>::template container<
    typename std::result_of<
      Function(typename new_executor_traits<Executor>::index_type)
    >::type
  >;

  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_default_container<
    Executor,
    Function,
    expected_return_type
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_returning_default_container(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::async_execute()


namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_async_execute_returning_void(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.async_execute(f, shape);
} // end multi_agent_async_execute_returning_void()


template<class Executor, class Function>
typename new_executor_traits<Executor>::template future<void>
  multi_agent_async_execute_returning_void(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto ready = new_executor_traits<Executor>::template make_ready_future<void>(ex);
  return new_executor_traits<Executor>::then_execute(ex, ready, f, shape);
} // end multi_agent_async_returning_default_void()


template<class Executor, class Function>
struct has_multi_agent_async_execute_returning_void_impl
{
  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>()
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

template<class Executor, class Function>
using has_multi_agent_async_execute_returning_void = typename has_multi_agent_async_execute_returning_void_impl<Executor,Function>::type;


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
typename new_executor_traits<Executor>::template future<void>
  new_executor_traits<Executor>
    ::async_execute(typename new_executor_traits<Executor>::executor_type& ex,
                    Function f,
                    typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_async_execute_returning_void<
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::multi_agent_async_execute_returning_void(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::async_execute()


} // end agency

