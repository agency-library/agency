#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_container.hpp>
#include <agency/future.hpp>
#include <agency/detail/invoke.hpp>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


struct return_unit
{
  __AGENCY_ANNOTATION
  unit operator()() const
  {
    return unit();
  }
};


template<class Function>
struct ignore_unit_result_parameter_and_invoke
{
  mutable Function f;

  // this is the case when the predecessor type is non-void
  template<class Index, class Predecessor, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }

  // this is the case when the predecessor type is void
  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, shared_parameters...);
  }
};


// this is the case for when Function returns void
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
executor_future_t<E,void>
  bulk_then_execute_with_auto_result(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  // wrap f in a functor that will ignore the unit object we pass to it
  ignore_unit_result_parameter_and_invoke<Function> g{f};

  // just call bulk_then_execute() and use a result factory that creates a unit object which can be easily discarded
  executor_future_t<E,unit> intermediate_future = bulk_then_execute(exec, g, shape, predecessor, return_unit(), factories...);

  // cast the intermediate_future to void
  // XXX we may wish to allow the executor to participate in this cast
  return future_traits<executor_future_t<E,unit>>::template cast<void>(intermediate_future);
}


template<class Executor, class T>
struct container_factory
{
  executor_shape_t<Executor> shape;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  executor_container_t<Executor,T> operator()() const
  {
    return executor_container_t<Executor,T>(shape);
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
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
executor_future_t<E,
  executor_container_t<E,
    result_of_continuation_t<Function,executor_index_t<E>,Future,result_of_t<Factories()>&...>
  >
>
  bulk_then_execute_with_auto_result(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  // compute the type of f's result
  using result_type = result_of_continuation_t<Function,executor_index_t<E>,Future,result_of_t<Factories()>...>;

  // compute the type of container that will store f's results
  using container_type = executor_container_t<E,result_type>;

  // wrap f in a functor that will store f's result
  invoke_and_store_result<Function, container_type> g{f};

  // call bulk_then_execute() and use a result factory that creates a container to store f's results
  return bulk_then_execute(exec, g, shape, predecessor, container_factory<E,result_type>{shape}, factories...);
}


} // end new_executor_traits_detail
} // end detail
} // end agency

