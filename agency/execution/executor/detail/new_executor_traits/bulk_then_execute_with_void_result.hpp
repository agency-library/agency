#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// this is the general case when the predecessor future is non-void
template<class Function, class Predecessor>
struct ignore_unit_result_parameter_and_invoke
{
  mutable Function f;

  // this is the case when the predecessor type is non-void
  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }
};


// this is specialization when the predecessor future is void
template<class Function>
struct ignore_unit_result_parameter_and_invoke<Function,void>
{
  mutable Function f;

  // this is the case when the predecessor type is void
  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, shared_parameters...);
  }
};


__agency_exec_check_disable__
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<E,void>
  bulk_then_execute_with_void_result(E& exec, Function f, new_executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  using predecessor_type = future_value_t<Future>;

  // wrap f in a functor that will ignore the unit object we pass to it
  ignore_unit_result_parameter_and_invoke<Function,predecessor_type> g{f};

  // just call bulk_then_execute() and use a result factory that creates a unit object which can be easily discarded
  executor_future_t<E,unit> intermediate_future = bulk_then_execute(exec, g, shape, predecessor, unit_factory(), factories...);

  // cast the intermediate_future to void
  // XXX we may wish to allow the executor to participate in this cast
  return future_traits<executor_future_t<E,unit>>::template cast<void>(intermediate_future);
}


} // end new_executor_traits_detail
} // end detail
} // end agency

