#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/new_executor_traits/executor_container.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_then_execute_with_void_result.hpp>
#include <agency/detail/invoke.hpp>


namespace agency
{
namespace detail
{
namespace executor_customization_points_detail
{


// this is the case for when Function returns void
__agency_exec_check_disable__
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(std::is_void<result_of_continuation_t<Function, new_executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
__AGENCY_ANNOTATION
new_executor_future_t<E,void>
  bulk_then_execute_with_auto_result(E& exec, Function f, new_executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  return bulk_then_execute_with_void_result(exec, f, shape, predecessor, factories...);
}


template<class Executor, class T>
struct container_factory
{
  new_executor_shape_t<Executor> shape;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  new_executor_container_t<Executor,T> operator()() const
  {
    return new_executor_container_t<Executor,T>(shape);
  }
};

template<class Function, class Container>
struct invoke_and_store_result
{
  mutable Function f;

  // this is the case when the predecessor type is non-void
  template<class Index, class Predecessor, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, Container& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }

  // this is the case when the predecessor type is void
  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Container& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, shared_parameters...);
  }
};


// this is the case for when Function returns non-void
// when Function does not return void, this function collects
// the results of each invocation into a container
// this container is returned through a future
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_continuation_t<Function, new_executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
__AGENCY_ANNOTATION
new_executor_future_t<E,
  new_executor_container_t<E,
    result_of_continuation_t<Function,new_executor_index_t<E>,Future,result_of_t<Factories()>&...>
  >
>
  bulk_then_execute_with_auto_result(E& exec, Function f, new_executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  // compute the type of f's result
  using result_type = result_of_continuation_t<Function,new_executor_index_t<E>,Future,result_of_t<Factories()>...>;

  // compute the type of container that will store f's results
  using container_type = new_executor_container_t<E,result_type>;

  // wrap f in a functor that will store f's result
  invoke_and_store_result<Function, container_type> g{f};

  // call bulk_then_execute() and use a result factory that creates a container to store f's results
  return bulk_then_execute(exec, g, shape, predecessor, container_factory<E,result_type>{shape}, factories...);
}


} // end executor_customization_points_detail
} // end detail
} // end agency

