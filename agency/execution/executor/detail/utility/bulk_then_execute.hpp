#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/adaptors/bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/type_traits.hpp>


namespace agency
{
namespace detail
{


// this case handles executors which may be adapted by detail::bulk_then_executor
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<ResultFactory()>
>
bulk_then_execute(const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  return detail::bulk_then_executor<E>(exec).bulk_then_execute(f, shape, predecessor, result_factory, shared_factories...);
}


} // end detail
} // end agency

