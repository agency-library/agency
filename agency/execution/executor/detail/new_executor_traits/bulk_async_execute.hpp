#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
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


template<class E, class Function, class Factory1, class Factory2,
         __AGENCY_REQUIRES(BulkAsynchronousExecutor<E>())
        >
executor_future_t<
  E,
  result_of_t<Factory1(executor_shape_t<E>)>
>
bulk_async_execute(E& exec, Function f, executor_shape_t<E> shape, Factory1 result_factory, Factory2 shared_factory)
{
  return exec.bulk_async_execute(f, shape, result_factory, shared_factory);
}


template<class E, class Function, class Factory1, class Factory2,
         __AGENCY_REQUIRES(BulkExecutor<E>() && !BulkAsynchronousExecutor<E>())
        >
executor_future_t<
  E,
  result_of_t<Factory1(executor_shape_t<E>)>
>
bulk_async_execute(E& exec, Function f, executor_shape_t<E> shape, Factory1 result_factory, Factory2 shared_factory)
{
  using void_future_type = executor_future_t<E,void>;

  // XXX we might want to actually allow the executor to participate here
  auto predecessor = future_traits<void_future_type>::make_ready();

  return bulk_then_execute(exec, f, shape, predecessor, result_factory, shared_factory);
}


} // end new_executor_traits_detail
} // end detail
} // end agency

