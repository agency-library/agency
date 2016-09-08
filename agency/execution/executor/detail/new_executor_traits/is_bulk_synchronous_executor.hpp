#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_execution_depth_or.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_shape_type_or.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_index_type_or.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function, class Shape,
         class ResultFactory,
         class... SharedFactories
        >
struct has_bulk_execute_impl
{
  using expected_return_type = result_of_t<ResultFactory()>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<Executor1>().bulk_execute(
               std::declval<Function>(),
               std::declval<Shape>(),
               std::declval<ResultFactory>(),
               std::declval<SharedFactories>()...
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


template<class Executor, class Function, class Shape,
         class ResultFactory,
         class... SharedFactories
        >
using has_bulk_execute = typename has_bulk_execute_impl<Executor, Function, Shape, ResultFactory, SharedFactories...>::type;


template<class T, class IndexSequence>
struct is_bulk_synchronous_executor_impl;

template<class T, size_t... Indices>
struct is_bulk_synchronous_executor_impl<T, index_sequence<Indices...>>
{
  // executor properties
  using shape_type = member_shape_type_or_t<T,size_t>;
  using index_type = member_index_type_or_t<T,shape_type>;

  // types related to functions passed to .bulk_execute()
  using result_type = int;
  template<size_t>
  using shared_type = int;

  // the functions we'll pass to .bulk_execute() to test
  
  // XXX WAR nvcc 8.0 bug
  //using test_function = std::function<void(index_type, result_type&, shared_type<Indices>&...)>;
  //using test_result_factory = std::function<result_type()>;

  struct test_function
  {
    void operator()(index_type, result_type&, shared_type<Indices>&...);
  };

  struct test_result_factory
  {
    result_type operator()();
  };

  // XXX WAR nvcc 8.0 bug
  //template<size_t I>
  //using test_shared_factory = std::function<shared_type<I>()>;

  template<size_t I>
  struct test_shared_factory
  {
    shared_type<I> operator()();
  };

  using type = has_bulk_execute<
    T,
    test_function,
    shape_type,
    test_result_factory,
    test_shared_factory<Indices>...
  >;
};

template<class T>
using is_bulk_synchronous_executor = typename is_bulk_synchronous_executor_impl<
  T,
  make_index_sequence<
    executor_execution_depth_or<T>::value
  >
>::type;


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool BulkSynchronousExecutor()
{
  return is_bulk_synchronous_executor<T>();
}


} // end new_executor_traits_detail
} // end detail
} // end agency

