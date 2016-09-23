#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_then_execute.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace executor_customization_points_detail
{


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
new_executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_async_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_async_execute(f, shape, result_factory, shared_factories...);
}


__agency_exec_check_disable__
template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>() && !BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
new_executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_async_execute(E& exec, Function f, new_executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  using void_future_type = new_executor_future_t<E,void>;

  // XXX we might want to actually allow the executor to participate here
  auto predecessor = future_traits<void_future_type>::make_ready();

  return bulk_then_execute(exec, f, shape, predecessor, result_factory, shared_factories...);
}


} // end executor_customization_points_detail
} // end detail
} // end agency

