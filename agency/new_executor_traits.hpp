#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


__DEFINE_HAS_NESTED_TYPE(has_execution_category, execution_category);
__DEFINE_HAS_NESTED_TYPE(has_index_type, index_type);
__DEFINE_HAS_NESTED_TYPE(has_shape_type, shape_type);


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


template<class Executor, class TupleOfFutures, class Function>
struct has_single_agent_when_all_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all_execute(
               std::declval<TupleOfFutures>(),
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class TupleOfFutures, class Function>
using has_single_agent_when_all_execute = typename has_single_agent_when_all_execute_impl<Executor, TupleOfFutures, Function>::type;


template<class Executor, class... Futures>
struct has_when_all_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().when_all(
               *std::declval<Futures*>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class... Futures>
using has_when_all = typename has_when_all_impl<Executor, Futures...>::type;


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

    template<class T, class... Args>
    static future<T> make_ready_future(executor_type& ex, Args&&... args);

    // single-agent when_all_execute_and_select()
    template<size_t... Indices, class TupleOfFutures, class Function>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, TupleOfFutures&& futures, Function f);

    // multi-agent when_all_execute_and_select()
    template<size_t... Indices, class TupleOfFutures, class Function>
    static future<
      detail::when_all_execute_and_select_result_t<
        detail::index_sequence<Indices...>,
        typename std::decay<TupleOfFutures>::type
      >
    >
      when_all_execute_and_select(executor_type& ex, TupleOfFutures&& futures, Function f, shape_type shape);

    // single-agent then_execute()
    template<class Future, class Function>
    static future<
      typename detail::future_result_of<Function,Future>::type
    > then_execute(executor_type& ex, Future& fut, Function f);
}; // end new_executor_traits


} // end agency

#include <agency/detail/executor_traits/make_ready_future.hpp>
#include <agency/detail/executor_traits/when_all_execute_and_select.hpp>
#include <agency/detail/executor_traits/then_execute.hpp>

