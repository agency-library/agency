#pragma once

#include <type_traits>
#include <agency/detail/config.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <agency/detail/executor_traits/member_types.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/future.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


struct test_function_returning_int
{
  template<class... Args>
  __AGENCY_ANNOTATION
  int operator()(Args&&...) { return 0; }
};


struct test_function_returning_void
{
  template<class... Args>
  __AGENCY_ANNOTATION
  void operator()(Args&&...) {}
};



template<class Executor, class T, class Future>
struct has_future_cast_impl
{
  template<
    class Executor2,
    typename = decltype(
      std::declval<Executor2*>()->template future_cast<T>(
        *std::declval<Future*>()
      )
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class T, class Future>
using has_future_cast = typename has_future_cast_impl<Executor,T,Future>::type;


template<class Executor, class T, class... Args>
struct has_make_ready_future_impl
{
  template<
    class Executor2,
    typename = decltype(
      std::declval<Executor2*>()->template make_ready_future<T>(
        std::declval<Args>()...
      )
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class T, class... Args>
using has_make_ready_future = typename has_make_ready_future_impl<Executor,T,Args...>::type;


template<class Executor>
struct has_shape_impl
{
  template<
    class Executor2,
    class = decltype(std::declval<Executor2>().shape())
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor>
using has_shape = typename has_shape_impl<Executor>::type;


template<class Executor>
struct has_max_shape_dimensions_impl
{
  template<
    class Executor2,
    class = decltype(
      std::declval<Executor2>().max_shape_dimensions()
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor>
using has_max_shape_dimensions = typename has_max_shape_dimensions_impl<Executor>::type;


template<class Executor, class Function>
struct has_multi_agent_async_execute_returning_default_container_impl
{
  using index_type           = executor_index_t<Executor>;
  using shape_type           = executor_shape_t<Executor>;
  using container_value_type = result_of_t<Function(index_type)>;
  using container_type       = executor_container_t<Executor,container_value_type>;
  using expected_return_type = executor_future_t<Executor, container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<shape_type>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_int>
using has_multi_agent_async_execute_returning_default_container = typename has_multi_agent_async_execute_returning_default_container_impl<Executor,Function>::type;


template<class Executor, class Function, class Factory>
struct has_multi_agent_async_execute_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using container_type = result_of_t<Factory(shape_type)>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory>
using has_multi_agent_async_execute_returning_user_specified_container = typename has_multi_agent_async_execute_returning_user_specified_container_impl<Executor,Function,Factory>::type;


template<class Executor, class Function>
struct has_multi_agent_async_execute_returning_void_impl
{
  using shape_type           = executor_shape_t<Executor>;
  using expected_return_type = executor_future_t<Executor,void>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<shape_type>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_void>
using has_multi_agent_async_execute_returning_void = typename has_multi_agent_async_execute_returning_void_impl<Executor,Function>::type;


template<class Executor, class Function, class... Factories>
struct has_multi_agent_async_execute_with_shared_inits_returning_default_container_impl
{
  using index_type           = executor_index_t<Executor>;
  using shape_type           = executor_shape_t<Executor>;
  using container_value_type = result_of_t<Function(index_type, result_of_t<Factories()>&...)>;
  using container_type       = executor_container_t<Executor,container_value_type>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class... Factories>
using has_multi_agent_async_execute_with_shared_inits_returning_default_container = typename has_multi_agent_async_execute_with_shared_inits_returning_default_container_impl<Executor,Function,Factories...>::type;


template<class Executor, class Function, class Factory, class... Factories>
struct has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using container_type = result_of_t<Factory(shape_type)>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory, class... Factories>
using has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container = typename has_multi_agent_async_execute_with_shared_inits_returning_user_specified_container_impl<Executor,Function,Factory,Factories...>::type;


template<class Executor, class Function, class... Types>
struct has_multi_agent_async_execute_with_shared_inits_returning_void_impl
{
  using index_type           = executor_index_t<Executor>;
  using shape_type           = executor_shape_t<Executor>;
  using expected_return_type = executor_future_t<Executor,void>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().async_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<Types>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class... Types>
using has_multi_agent_async_execute_with_shared_inits_returning_void = typename has_multi_agent_async_execute_with_shared_inits_returning_void_impl<Executor,Function,Types...>::type;


template<class Executor, class Function>
struct has_multi_agent_execute_returning_default_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using index_type = executor_index_t<Executor>;
  using container_value_type = result_of_t<Function(index_type)>;
  using expected_return_type = executor_container_t<Executor,container_value_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<shape_type>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_int>
using has_multi_agent_execute_returning_default_container = typename has_multi_agent_execute_returning_default_container_impl<Executor,Function>::type;


template<class Executor, class Function, class Factory>
struct has_multi_agent_execute_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using expected_return_type = result_of_t<Factory(shape_type)>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory>
using has_multi_agent_execute_returning_user_specified_container = typename has_multi_agent_execute_returning_user_specified_container_impl<Executor,Function,Factory>::type;


template<class Executor, class Function>
struct has_multi_agent_execute_returning_void_impl
{
  using shape_type = executor_shape_t<Executor>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<shape_type>()
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

template<class Executor, class Function = test_function_returning_void>
using has_multi_agent_execute_returning_void = typename has_multi_agent_execute_returning_void_impl<Executor,Function>::type;


template<class Executor, class Function, class Factory, class... Factories>
struct has_multi_agent_execute_with_shared_inits_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using expected_return_type = result_of_t<Factory(shape_type)>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory, class... Factories>
using has_multi_agent_execute_with_shared_inits_returning_user_specified_container = typename has_multi_agent_execute_with_shared_inits_returning_user_specified_container_impl<Executor,Function,Factory,Factories...>::type;


template<class Executor, class Function, class... Factories>
struct has_multi_agent_execute_with_shared_inits_returning_default_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using index_type = executor_index_t<Executor>;
  using container_value_type = result_of_t<Function(index_type, result_of_t<Factories()>&...)>;
  using expected_return_type = executor_container_t<Executor,container_value_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class... Factories>
using has_multi_agent_execute_with_shared_inits_returning_default_container = typename has_multi_agent_execute_with_shared_inits_returning_default_container_impl<Executor,Function,Factories...>::type;


template<class Executor, class Function, class... Types>
struct has_multi_agent_execute_with_shared_inits_returning_void_impl
{
  using shape_type = executor_shape_t<Executor>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<Types>()...
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

template<class Executor, class Function, class... Types>
using has_multi_agent_execute_with_shared_inits_returning_void = typename has_multi_agent_execute_with_shared_inits_returning_void_impl<Executor,Function,Types...>::type;


template<class Executor, class TypeList>
struct has_any_multi_agent_execute_impl;


template<class Executor, class... Factories>
struct has_any_multi_agent_execute_impl<Executor, type_list<Factories...>>
{
  // XXX this implementation is incomplete -- there are more variants
  //     of multi-agent execute() than are checked for here

  using has_execute_with_shared_inits_returning_user_specified_container = has_multi_agent_execute_with_shared_inits_returning_user_specified_container<
    Executor,
    test_function_returning_int,
    container_factory<executor_traits_detail::discarding_container>,
    Factories...
  >;

  using has_execute_with_shared_inits_returning_void = has_multi_agent_execute_with_shared_inits_returning_void<
    Executor,
    test_function_returning_void,
    Factories...
  >;

  using has_execute_returning_void = has_multi_agent_execute_returning_void<
    Executor,
    test_function_returning_void
  >;

  using type = detail::disjunction<
    has_execute_with_shared_inits_returning_user_specified_container,
    has_execute_with_shared_inits_returning_void,
    has_execute_returning_void
  >;
};


template<class T>
struct has_any_multi_agent_execute
  : has_any_multi_agent_execute_impl<
      T,
      type_list_repeat<
        execution_depth<typename T::execution_category>::value, unit_factory
      >
    >::type
{};


template<class Executor, class Function, class Future>
struct has_multi_agent_then_execute_returning_default_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using index_type = executor_index_t<Executor>;
  using container_value_type = typename detail::result_of_continuation<Function,index_type,Future>::type;
  using container_type       = executor_container_t<Executor,container_value_type>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_int, class Future = executor_future_t<Executor,int>>
using has_multi_agent_then_execute_returning_default_container = typename has_multi_agent_then_execute_returning_default_container_impl<Executor,Function,Future>::type;


template<class Executor, class Function, class Factory, class Future>
struct has_multi_agent_then_execute_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using container_type = result_of_t<Factory(shape_type)>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory, class Future>
using has_multi_agent_then_execute_returning_user_specified_container = typename has_multi_agent_then_execute_returning_user_specified_container_impl<Executor,Function,Factory,Future>::type;


template<class Executor, class Function, class Future>
struct has_multi_agent_then_execute_returning_void_impl
{
  using shape_type = executor_shape_t<Executor>;
  using expected_return_type = executor_future_t<Executor,void>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               *std::declval<Future*>()
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_void, class Future = executor_future_t<Executor,int>>
using has_multi_agent_then_execute_returning_void = typename has_multi_agent_then_execute_returning_void_impl<Executor,Function,Future>::type;


template<class Executor, class Function, class Future, class... Factories>
struct has_multi_agent_then_execute_with_shared_inits_returning_default_container_impl
{
  using shape_type           = executor_shape_t<Executor>;
  using index_type           = executor_index_t<Executor>;
  using container_value_type = typename detail::result_of_continuation<Function,index_type,Future,result_of_t<Factories()>&...>::type;
  using container_type       = executor_container_t<Executor,container_value_type>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               *std::declval<Future*>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType, expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Future, class... Factories>
using has_multi_agent_then_execute_with_shared_inits_returning_default_container = typename has_multi_agent_then_execute_with_shared_inits_returning_default_container_impl<Executor,Function,Future,Factories...>::type;


template<class Executor, class Function, class Factory, class Future, class... Factories>
struct has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_impl
{
  using shape_type = executor_shape_t<Executor>;
  using container_type = result_of_t<Factory(shape_type)>;
  using expected_return_type = executor_future_t<Executor,container_type>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<Factory>(),
               std::declval<shape_type>(),
               *std::declval<Future*>(),
               std::declval<Factories>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Factory, class Future, class... Factories>
using has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container = typename has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container_impl<Executor,Function,Factory,Future,Factories...>::type;


template<class Executor, class Function, class Future, class... Types>
struct has_multi_agent_then_execute_with_shared_inits_returning_void_impl
{
  using shape_type = executor_shape_t<Executor>;
  using expected_return_type = executor_future_t<Executor,void>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().then_execute(
               std::declval<Function>(),
               std::declval<shape_type>(),
               *std::declval<Future*>(),
               std::declval<Types>()...
             )
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_return_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class Future, class... Types>
using has_multi_agent_then_execute_with_shared_inits_returning_void = typename has_multi_agent_then_execute_with_shared_inits_returning_void_impl<Executor,Function,Future,Types...>::type;


template<class Executor, class T, class TypeList>
struct has_any_multi_agent_then_execute_impl;


template<class Executor, class T, class... Factories>
struct has_any_multi_agent_then_execute_impl<Executor, T, type_list<Factories...>>
{
  // XXX this implementation is incomplete -- there are more variants
  //     of multi-agent then_execute() than are checked for here

  using has_then_execute_with_shared_inits_returning_user_specified_container = has_multi_agent_then_execute_with_shared_inits_returning_user_specified_container<
    Executor,
    test_function_returning_int,
    container_factory<executor_traits_detail::discarding_container>,
    executor_future_t<Executor,void>,
    Factories...
  >;

  using has_then_execute_with_shared_inits_returning_void = has_multi_agent_then_execute_with_shared_inits_returning_void<
    Executor,
    test_function_returning_void,
    executor_future_t<Executor,void>,
    Factories...
  >;

  using has_then_execute_returning_void = has_multi_agent_then_execute_returning_void<
    Executor,
    test_function_returning_void,
    executor_future_t<Executor,void>
  >;

  using type = detail::disjunction<
    has_then_execute_with_shared_inits_returning_user_specified_container,
    has_then_execute_with_shared_inits_returning_void
  >;
};


template<class T>
struct has_any_multi_agent_then_execute
  : has_any_multi_agent_then_execute_impl<
      T,
      int,
      type_list_repeat<
        execution_depth<typename T::execution_category>::value, unit_factory
      >
    >::type
{};


template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
struct has_multi_agent_when_all_execute_and_select_impl
{
  using shape_type = executor_shape_t<Executor>;

  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<TupleOfFutures>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
using has_multi_agent_when_all_execute_and_select = typename has_multi_agent_when_all_execute_and_select_impl<Executor, Function, TupleOfFutures, Indices...>::type;


template<class IndexSequence, class Executor, class Function, class TupleOfFutures, class TypeList>
struct has_multi_agent_when_all_execute_and_select_with_shared_inits_impl;


template<size_t... Indices, class Executor, class Function, class TupleOfFutures, class... Types>
struct has_multi_agent_when_all_execute_and_select_with_shared_inits_impl<index_sequence<Indices...>, Executor, Function, TupleOfFutures, type_list<Types...>>
{
  using shape_type = executor_shape_t<Executor>;

  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<Function>(),
               std::declval<shape_type>(),
               std::declval<TupleOfFutures>(),
               std::declval<Types>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class IndexSequence, class Executor, class Function, class TupleOfFutures, class TypeList>
using has_multi_agent_when_all_execute_and_select_with_shared_inits = typename has_multi_agent_when_all_execute_and_select_with_shared_inits_impl<IndexSequence, Executor, Function, TupleOfFutures, TypeList>::type;


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
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_int>
using has_single_agent_async_execute = typename has_single_agent_async_execute_impl<Executor,Function>::type;


template<class Executor, class Function>
struct has_single_agent_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function = test_function_returning_int>
using has_single_agent_execute = typename has_single_agent_execute_impl<Executor,Function>::type;


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

template<class Executor, class Function = test_function_returning_int, class Future = executor_future_t<Executor,void>>
using has_single_agent_then_execute = typename has_single_agent_then_execute_impl<Executor,Function,Future>::type;


template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
struct has_single_agent_when_all_execute_and_select_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().template when_all_execute_and_select<Indices...>(
               std::declval<Function>(),
               std::declval<TupleOfFutures>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class TupleOfFutures, size_t... Indices>
using has_single_agent_when_all_execute_and_select = typename has_single_agent_when_all_execute_and_select_impl<Executor, Function, TupleOfFutures, Indices...>::type;


template<class Executor, class... Futures>
struct has_when_all_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all(
               std::declval<Futures>()...
             )
           )
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class... Futures>
using has_when_all = typename has_when_all_impl<Executor,Futures...>::type;


template<class Executor, class Function, class TupleOfFutures>
struct has_single_agent_when_all_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all_execute(
               std::declval<Function>(),
               std::declval<TupleOfFutures>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function, class TupleOfFutures>
using has_single_agent_when_all_execute = typename has_single_agent_when_all_execute_impl<Executor, Function, TupleOfFutures>::type;


} // end executor_traits_detail
} // end detail
} // end agency

