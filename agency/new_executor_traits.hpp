#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/detail/type_traits.hpp>
#include <vector>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


__DEFINE_HAS_NESTED_TYPE(has_execution_category, execution_category);
__DEFINE_HAS_NESTED_TYPE(has_index_type, index_type);
__DEFINE_HAS_NESTED_TYPE(has_shape_type, shape_type);
__DEFINE_HAS_NESTED_CLASS_TEMPLATE(has_container_template, container);


template<class T, class Index>
struct is_container_impl
{
  // test if T is a container by trying to index it using the bracket operator
  // XXX should also check that it is constructible from Shape
  template<class T1,
           class Index1 = Index,
           class Reference = decltype(
             (*std::declval<T1*>())[std::declval<Index1>()]
           ),
           class = typename std::enable_if<
             !std::is_void<Reference>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T, class Index>
using is_container = typename is_container_impl<T,Index>::type;


template<class T>
struct nested_execution_category
{
  using type = typename T::execution_category;
};


template<class T, class Default = parallel_execution_tag>
struct nested_execution_category_with_default
  : agency::detail::lazy_conditional<
      has_execution_category<T>::value,
      nested_execution_category<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct nested_index_type
{
  using type = typename T::index_type;
};


template<class T, class Default = size_t>
struct nested_index_type_with_default
  : agency::detail::lazy_conditional<
      has_index_type<T>::value,
      nested_index_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct nested_shape_type
{
  using type = typename T::shape_type;
};


template<class T, class Default = size_t>
struct nested_shape_type_with_default
  : agency::detail::lazy_conditional<
      has_shape_type<T>::value,
      nested_shape_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T, class U>
struct has_future_impl
{
  template<class> static std::false_type test(...);
  template<class X> static std::true_type test(typename X::template future<U>* = 0);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_future = typename has_future_impl<T,U>::type;


template<class Executor, class T, bool = has_future<Executor,T>::value>
struct executor_future
{
  using type = typename Executor::template future<T>;
};

template<class Executor, class T>
struct executor_future<Executor,T,false>
{
  using type = std::future<T>;
};


//template<class T>
//struct nested_container_template
//{
//  template<class U>
//  using type_template = typename T::template container<U>;
//};

template<class T, class U>
using member_container_t = typename T::template container<U>;

// XXX WAR problems with gcc 4.X and nvcc 7.5
//template<class Default, class T, class U>
//using member_container_or_t = detected_or_t<Default, member_container_t, T, U>;

template<class T, class U>
struct has_member_container_template_impl
{
  template<class T1,
           class U1 = U,
           class = typename T1::template container<U>
          >
  static std::true_type test(int);
           
  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};

template<class T, class U>
using has_member_container_template = typename has_member_container_template_impl<T,U>::type;

template<class T, class U, bool = has_member_container_template<T,U>::value>
struct member_container
{
};

template<class T, class U>
struct member_container<T,U,true>
{
  using type = typename T::template container<U>;
};

template<class Default, class T, class U>
using member_container_or_t = typename lazy_conditional<
  has_member_container_template<T,U>::value,
  member_container<T,U>,
  identity<Default>
>::type;


template<bool condition, class Then, class Else>
struct lazy_conditional_template
{
  template<class... T>
  using type_template = typename Then::template type_template<T...>;
};

template<class Then, class Else>
struct lazy_conditional_template<false, Then, Else>
{
  template<class... T>
  using type_template = typename Else::template type_template<T...>;
};


template<template<class...> class T>
struct identity_template
{
  template<class... U>
  using type_template = T<U...>;
};


//template<class T, template<class,class> class Default = std::vector>
//struct nested_container_with_default
//  : lazy_conditional_template<
//      has_container_template<T,int>::value,
//      nested_container_template<T>,
//      identity_template<Default>
//    >
//{};


template<class Default, class T, class U>
struct member_container_or
{
};


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


} // end new_executor_traits_detail
} // end detail


template<class Executor>
struct new_executor_traits
{
  public:
    using executor_type = Executor;

    using execution_category = typename detail::new_executor_traits_detail::nested_execution_category_with_default<
      executor_type,
      parallel_execution_tag
    >::type;

    constexpr static size_t execution_depth = detail::execution_depth<execution_category>::value;

    using index_type = typename detail::new_executor_traits_detail::nested_index_type_with_default<
      executor_type,
      size_t
    >::type;

    using shape_type = typename detail::new_executor_traits_detail::nested_shape_type_with_default<
      executor_type,
      index_type
    >::type;

    template<class T>
    using future = typename detail::new_executor_traits_detail::executor_future<executor_type,T>::type;

    //template<class T>
    //using container = typename detail::new_executor_traits_detail::nested_container_with_default<
    //  executor_type,
    //  std::vector
    //>::template type_template<T>;
    template<class T>
    using container = detail::new_executor_traits_detail::member_container_or_t<std::vector<T>, executor_type, T>;

    template<class T, class... Args>
    __AGENCY_ANNOTATION
    static future<T> make_ready_future(executor_type& ex, Args&&... args);

    template<class T, class Future>
    __AGENCY_ANNOTATION
    static future<T> future_cast(executor_type& ex, Future& fut);

    template<class... Futures>
    static future<
      detail::when_all_result_t<typename std::decay<Futures>::type...>
    > when_all(executor_type& ex, Futures&&... futures);

    // single-agent when_all_execute_and_select()
    template<size_t... Indices, class Function, class TupleOfFutures>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function f, TupleOfFutures&& futures);

    // multi-agent when_all_execute_and_select()
    template<size_t... Indices, class Function, class TupleOfFutures>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function f, shape_type shape, TupleOfFutures&& futures);

    // multi-agent when_all_execute_and_select() with shared parameters
    template<size_t... Indices, class Function, class TupleOfFutures, class Factory, class... Factories>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, Function f, shape_type shape, TupleOfFutures&& futures, Factory outer_shared_factory, Factories... inner_shared_factories);

    // single-agent then_execute()
    template<class Function, class Future>
    __AGENCY_ANNOTATION
    static future<
      detail::result_of_continuation_t<Function,Future>
    >
      then_execute(executor_type& ex, Function f, Future& fut);

    // multi-agent then_execute() returning user-specified Container
    template<class Container, class Function, class Future,
             class = typename std::enable_if<
               detail::new_executor_traits_detail::is_container<Container,index_type>::value
             >::type,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = detail::result_of_continuation_t<
               Function,
               index_type,
               Future
             >
            >
    __AGENCY_ANNOTATION
    static future<Container> then_execute(executor_type& ex, Function f, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning user-specified Container
    template<class Container, class Function, class Future, class... Factories,
             class = typename std::enable_if<
               detail::new_executor_traits_detail::is_container<Container,index_type>::value
             >::type,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = detail::result_of_continuation_t<
               Function,
               index_type,
               Future,
               typename std::result_of<Factories()>::type&...
             >>
    __AGENCY_ANNOTATION
    static future<Container> then_execute(executor_type& ex, Function f, shape_type shape, Future& fut, Factories... shared_factories);

    // multi-agent then_execute() returning default container
    template<class Function, class Future,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future>::value
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future>
               >::value
             >::type>
    static future<
      container<
        detail::result_of_continuation_t<Function,index_type,Future>
      >
    >
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning default container
    template<class Function, class Future, class... Factories,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future,typename std::result_of<Factories()>::type&...>::value
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
               >::value
             >::type>
    static future<
      container<
        detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
      >
    >
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut, Factories... shared_factories);

    // multi-agent then_execute() returning void
    template<class Function, class Future,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future>::value
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future>
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static future<void>
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut);

    // multi-agent then_execute() with shared inits returning void
    template<class Function, class Future, class... Factories,
             class = typename std::enable_if<
               detail::is_future<Future>::value
             >::type,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               detail::is_callable_continuation<Function,index_type,Future,typename std::result_of<Factories()>::type&...>::value
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 detail::result_of_continuation_t<Function,index_type,Future,typename std::result_of<Factories()>::type&...>
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static future<void>
      then_execute(executor_type& ex, Function f, shape_type shape, Future& fut, Factories... shared_factories);

    // single-agent async_execute()
    template<class Function>
    static future<
      typename std::result_of<Function()>::type
    >
      async_execute(executor_type& ex, Function f);

    // multi-agent async_execute() returning user-specified Container
    template<class Container, class Function>
    static future<Container> async_execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent async_execute() with shared inits returning user-specified Container
    template<class Container, class Function, class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type>
    static future<Container> async_execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent async_execute() returning default container
    template<class Function,
             class = typename std::enable_if<
               !std::is_void<
                typename std::result_of<Function(index_type)>::type
               >::value
             >::type>
    static future<
      container<
        typename std::result_of<Function(index_type)>::type
      >
    >
      async_execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent async_execute() with shared inits returning default container
    template<class Function,
             class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
               >::value
             >::type>
    static future<
      container<
        typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
      >
    >
      async_execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent async_execute() returning void
    template<class Function,
             class = typename std::enable_if<
              std::is_void<
                typename std::result_of<Function(index_type)>::type
              >::value
             >::type>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent async_execute() with shared inits returning void
    template<class Function,
             class... Factories,
             class = typename std::enable_if<
               sizeof...(Factories) == execution_depth
             >::type,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<Function(index_type, typename std::result_of<Factories()>::type&...)>::type
               >::value
             >::type>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // single-agent execute()
    template<class Function>
    static typename std::result_of<Function()>::type
      execute(executor_type& ex, Function f);

    // multi-agent execute returning user-specified Container
    template<class Container, class Function>
    __AGENCY_ANNOTATION
    static Container execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent execute with shared inits returning user-specified Container
    template<class Container, class Function, class... Factories,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    __AGENCY_ANNOTATION
    static Container execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent execute returning default container
    template<class Function,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<
                   Function(index_type)
                 >::type
               >::value
             >::type>
    static container<
      typename std::result_of<Function(index_type)>::type
    >
      execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent execute with shared inits returning default container
    template<class Function, class... Factories,
             class = typename std::enable_if<
               !std::is_void<
                 typename std::result_of<
                   Function(index_type, typename std::result_of<Factories()>::type&...)
                 >::type
               >::value
             >::type,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    static container<
      typename std::result_of<
        Function(
          index_type,
          typename std::result_of<Factories()>::type&...
        )
      >::type
    >
      execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);

    // multi-agent execute returning void
    template<class Function,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<
                   Function(index_type)
                 >::type
               >::value
             >::type>
    __AGENCY_ANNOTATION
    static void execute(executor_type& ex, Function f, shape_type shape);

    // multi-agent execute with shared inits returning void
    template<class Function, class... Factories,
             class = typename std::enable_if<
               std::is_void<
                 typename std::result_of<
                   Function(index_type, typename std::result_of<Factories()>::type&...)
                 >::type
               >::value
             >::type,
             class = typename std::enable_if<
               execution_depth == sizeof...(Factories)
             >::type>
    __AGENCY_ANNOTATION
    static void execute(executor_type& ex, Function f, shape_type shape, Factories... shared_factories);
}; // end new_executor_traits


} // end agency

#include <agency/detail/executor_traits/make_ready_future.hpp>
#include <agency/detail/executor_traits/future_cast.hpp>
#include <agency/detail/executor_traits/single_agent_when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/multi_agent_when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/multi_agent_when_all_execute_and_select_with_shared_inits.hpp>
#include <agency/detail/executor_traits/single_agent_then_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_then_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/single_agent_async_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_async_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/single_agent_execute.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_returning_void.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_user_specified_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_default_container.hpp>
#include <agency/detail/executor_traits/multi_agent_execute_with_shared_inits_returning_void.hpp>
#include <agency/detail/executor_traits/when_all.hpp>

