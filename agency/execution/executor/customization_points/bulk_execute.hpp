#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_async_execute.hpp>

namespace agency
{


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(detail::BulkSynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
detail::result_of_t<ResultFactory()>
bulk_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_execute(f, shape, result_factory, shared_factories...);
}


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(detail::BulkExecutor<E>() && !detail::BulkSynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
detail::result_of_t<ResultFactory()>
bulk_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return agency::bulk_async_execute(exec, f, shape, result_factory, shared_factories...).get();
}


} // end agency

