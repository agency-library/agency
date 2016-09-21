#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_index.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace bulk_execute_with_void_result_detail
{


template<class Function>
struct ignore_unit_result_parameter_and_invoke
{
  mutable Function f;

  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, shared_parameters...);
  }
};


} // end bulk_execute_with_void_result_detail


__agency_exec_check_disable__
template<class E, class Function, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
void bulk_execute_with_void_result(E& exec, Function f, executor_shape_t<E> shape, Factories... factories)
{
  namespace ns = bulk_execute_with_void_result_detail;

  // wrap f in a functor that will ignore the unit object we pass to it
  ns::ignore_unit_result_parameter_and_invoke<Function> g{f};

  // just call bulk_execute() and use a result factory that creates a unit object which can be easily discarded
  bulk_execute(exec, g, shape, unit_factory(), factories...);
}


} // end new_executor_traits_detail
} // end detail
} // end agency

