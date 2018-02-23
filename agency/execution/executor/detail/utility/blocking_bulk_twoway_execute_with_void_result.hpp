#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute.hpp>
#include <agency/execution/executor/detail/utility/invoke_functors.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>


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
void blocking_bulk_twoway_execute_with_void_result(const E& exec, Function f, executor_shape_t<E> shape, Factories... factories)
{
  // wrap f in a functor that will ignore the unit object we pass to it
  ignore_unit_result_parameter_and_invoke<Function> g{f};

  // just call blocking_bulk_twoway_execute() and use a result factory that creates a unit object which can be easily discarded
  detail::blocking_bulk_twoway_execute(exec, g, shape, unit_factory(), factories...);
}


} // end detail
} // end agency

