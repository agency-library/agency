#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_async_execute(E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_async_execute(f, shape, result_factory, shared_factories...);
}


template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>() && !BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_async_execute(E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  using void_future_type = executor_future_t<E,void>;

  // XXX we might want to actually allow the executor to participate here
  auto predecessor = future_traits<void_future_type>::make_ready();

  return bulk_then_execute(exec, f, shape, predecessor, result_factory, shared_factories...);
}


} // end new_executor_traits_detail
} // end detail
} // end agency

