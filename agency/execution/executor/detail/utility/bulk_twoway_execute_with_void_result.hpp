#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/customization_points/future_cast.hpp>
#include <agency/execution/executor/detail/utility/invoke_functors.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/properties/bulk.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/require.hpp>
#include <agency/detail/factory.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class E, class Function, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<E,void> bulk_twoway_execute_with_void_result(const E& exec, Function f, executor_shape_t<E> shape, Factories... factories)
{
  // wrap f in a functor that will ignore the unit object we pass to it
  ignore_unit_result_parameter_and_invoke<Function> g{f};

  // get a reference to avoid copies inside agency::require
  executor_ref<E> exec_ref{exec};

  // just call bulk_twoway_execute() and use a result factory that creates a unit object which can be easily discarded
  executor_future_t<E,unit> intermediate_future = agency::require(exec_ref, bulk, twoway).bulk_twoway_execute(g, shape, unit_factory(), factories...);

  // cast the intermediate_future to void
  return agency::future_cast<void>(exec, intermediate_future);
}


} // end detail
} // end agency

