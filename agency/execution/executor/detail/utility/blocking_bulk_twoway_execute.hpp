#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>
#include <agency/execution/executor/properties/bulk.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/require.hpp>


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
  // grab a reference to exec so that a copy is not created inside of require
  detail::executor_ref<E> exec_ref(exec);

  return agency::require(exec_ref, always_blocking, bulk, twoway).bulk_twoway_execute(f, shape, result_factory, shared_factories...).get();
}


} // end detail
} // end agency

