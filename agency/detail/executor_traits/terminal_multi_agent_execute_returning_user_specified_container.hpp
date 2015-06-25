#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>


// this is a version of multi-agent execute() returning user-specified container which does not attempt to recurse to async_execute() + wait()


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace terminal_multi_agent_execute_returning_user_specified_container_implementation_strategies
{


struct empty {};


template<class Container, class Function>
struct invoke_and_assign_result_to_container
{
  Container &c;
  mutable Function f;

  template<class Index>
  __AGENCY_ANNOTATION
  empty operator()(const Index& idx) const
  {
    // XXX should use std::invoke(f, idx);
    c[idx] = f(idx);

    // return something cheap to allocate
    return empty();
  }
};


struct use_multi_agent_execute_returning_user_specified_container_member_function {};

struct use_multi_agent_async_execute_returning_user_specified_container_member_function {};

struct use_multi_agent_execute_returning_void_member_function {};

struct use_multi_agent_execute_returning_default_container_member_function {};

struct use_multi_agent_async_execute_returning_default_container_member_function {};

struct use_for_loop {};


template<class Container, class Executor, class Function>
struct has_multi_agent_execute_returning_default_container_impl
{
  using type = new_executor_traits_detail::has_multi_agent_execute_returning_default_container<
    Executor,
    invoke_and_assign_result_to_container<Container, Function>
  >;
};


template<class Container, class Executor, class Function>
using has_multi_agent_execute_returning_default_container = typename has_multi_agent_execute_returning_default_container_impl<Container, Executor, Function>::type;


template<class Container, class Executor, class Function>
struct has_multi_agent_async_execute_returning_default_container_impl
{
  using type = new_executor_traits_detail::has_multi_agent_async_execute_returning_default_container<
    Executor,
    invoke_and_assign_result_to_container<Container, Function>
  >;
};


template<class Container, class Executor, class Function>
using has_multi_agent_async_execute_returning_default_container = typename has_multi_agent_async_execute_returning_default_container_impl<Container, Executor, Function>::type;


template<class Container, class Executor, class Function>
using select_multi_agent_terminal_execute_returning_user_specified_container_implementation = 
  typename std::conditional<
    has_multi_agent_execute_returning_user_specified_container<Container,Executor,Function>::value,
    use_multi_agent_execute_returning_user_specified_container_member_function,
    typename std::conditional<
      has_multi_agent_async_execute_returning_user_specified_container<Container,Executor,Function>::value,
      use_multi_agent_async_execute_returning_user_specified_container_member_function,
      typename std::conditional<
        has_multi_agent_execute_returning_void<Executor,Function>::value,
        use_multi_agent_execute_returning_void_member_function,
        typename std::conditional<
          has_multi_agent_execute_returning_default_container<Container,Executor,Function>::value,
          use_multi_agent_execute_returning_default_container_member_function,
          typename std::conditional<
            has_multi_agent_async_execute_returning_default_container<Container,Executor,Function>::value,
            use_multi_agent_async_execute_returning_default_container_member_function,
            use_for_loop
          >::type
        >::type
      >::type
    >::type
  >::type;


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_multi_agent_execute_returning_user_specified_container_member_function,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.template execute<Container>(f, shape);
} // end terminal_multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_multi_agent_async_execute_returning_user_specified_container_member_function,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto fut = ex.template async_execute<Container>(f, shape);

  // XXX should use an executor_traits operation on the future rather than .get()
  return fut.get();
} // end terminal_multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_multi_agent_execute_returning_void_member_function,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  Container result(shape);

  using index_type = typename new_executor_traits<Executor>::index_type;

  ex.execute([=,&result](const index_type& idx)
  {
    result[idx] = f(idx);
  },
  shape);

  return result;
} // end terminal_multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_multi_agent_execute_returning_default_container_member_function,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  // XXX the alternative to this implementation would be to do a conversion to the user's preferred container
  //     that alternative would allocate 2 * shape * sizeof(result_type) allocations
  //     the current implementation allocates shape * (sizeof(empty) + sizeof(result_type))
  
  Container result(shape);

  // ignore the container returned by this call
  ex.execute(invoke_and_assign_result_to_container<Container,Function>{result,f}, shape);

  return result;
} // end terminal_multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_multi_agent_async_execute_returning_default_container_member_function,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  // XXX the alternative to this implementation would be to do a conversion to the user's preferred container
  //     that alternative would allocate 2 * shape * sizeof(result_type) allocations
  //     the current implementation allocates shape * (sizeof(empty) + sizeof(result_type))

  Container result(shape);

  // ignore the container returned by this call
  ex.async_execute(invoke_and_assign_result_to_container<Container,Function>{result,f}, shape).wait();

  return result;
} // end terminal_multi_agent_execute_returning_user_specified_container()


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(use_for_loop,
                                                                          Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  Container result(shape);

  using index_type = typename new_executor_traits<Executor>::index_type;

  // XXX generalize to multidimensions or just use sequential_executor
  for(index_type idx = 0; idx < shape; ++idx)
  {
    result[idx] = f(idx);
  }

  return result;
} // end multi_agent_execute_returning_user_specified_container()


} // end terminal_multi_agent_execute_returning_user_specified_container_implementation_strategies


template<class Container, class Executor, class Function>
Container terminal_multi_agent_execute_returning_user_specified_container(Executor& ex,
                                                                          Function f,
                                                                          typename new_executor_traits<Executor>::shape_type shape)
{
  namespace ns = detail::new_executor_traits_detail::terminal_multi_agent_execute_returning_user_specified_container_implementation_strategies;

  using implementation_strategy = ns::select_multi_agent_terminal_execute_returning_user_specified_container_implementation<
    Container,
    Executor,
    Function
  >;

  return ns::terminal_multi_agent_execute_returning_user_specified_container<Container>(implementation_strategy(), ex, f, shape);
} // end terminal_multi_agent_execute_returning_user_specified_container()


} // end new_executor_traits_detail
} // end detail
} // end agency


