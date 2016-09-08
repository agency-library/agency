#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class E, class Function, class Factory1, class Factory2,
         __AGENCY_REQUIRES(BulkSynchronousExecutor<E>())
        >
result_of_t<Factory1()>
bulk_execute(E& exec, Function f, executor_shape_t<E> shape, Factory1 result_factory, Factory2 shared_factory)
{
  return exec.bulk_execute(f, shape, result_factory, shared_factory);
}


template<class E, class Function, class Factory1, class Factory2,
         __AGENCY_REQUIRES(BulkExecutor<E>() && !BulkSynchronousExecutor<E>())
        >
result_of_t<Factory1()>
bulk_execute(E& exec, Function f, executor_shape_t<E> shape, Factory1 result_factory, Factory2 shared_factory)
{
  return bulk_async_execute(exec, f, shape, result_factory, shared_factory).get();
}


} // end new_executor_traits_detail
} // end detail
} // end agency

