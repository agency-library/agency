#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_async_execute.hpp>

namespace agency
{
namespace detail
{
namespace executor_customization_points_detail
{


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkSynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
result_of_t<ResultFactory()>
bulk_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_execute(f, shape, result_factory, shared_factories...);
}


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>() && !BulkSynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
result_of_t<ResultFactory()>
bulk_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return bulk_async_execute(exec, f, shape, result_factory, shared_factories...).get();
}


} // end executor_customization_points_detail
} // end detail
} // end agency

