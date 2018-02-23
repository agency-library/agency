#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/bulk_twoway_execute_via_bulk_then_execute.hpp>


namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class Executor, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(is_bulk_twoway_executor<Executor>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<ResultFactory()>>
  bulk_twoway_execute(const Executor& ex, Function f, executor_shape_t<Executor> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return ex.bulk_twoway_execute(f, shape, result_factory, shared_factories...);
}

template<class Executor, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(
           !is_bulk_twoway_executor<Executor>::value and
           is_bulk_then_executor<Executor>::value
         )
        >
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<ResultFactory()>>
  bulk_twoway_execute(const Executor& ex, Function f, executor_shape_t<Executor> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return detail::bulk_twoway_execute_via_bulk_then_execute(ex, f, shape, result_factory, shared_factories...);
}

// XXX consider introducing an adaptation for bulk oneway executors here

// XXX consider introducing adaptations for non-bulk executors here


} // end detail
} // end agency

