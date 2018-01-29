#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_async_execute.hpp>
#include <agency/execution/executor/detail/adaptors/always_blocking_executor.hpp>

namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
detail::result_of_t<ResultFactory()>
blocking_bulk_twoway_execute(const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  detail::always_blocking_executor<E> blocking_exec(exec);
  return agency::bulk_async_execute(blocking_exec, f, shape, result_factory, shared_factories...).get();
}


} // end detail
} // end agency

