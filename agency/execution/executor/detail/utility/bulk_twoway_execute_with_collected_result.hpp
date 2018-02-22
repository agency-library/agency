#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/utility/invoke_functors.hpp>
#include <agency/execution/executor/properties/bulk.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/require.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class E, class Function, class ResultFactory, class... SharedFactories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(SharedFactories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_t<Function(executor_index_t<E>, result_of_t<SharedFactories()>&...)>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,result_of_t<ResultFactory()>>
  bulk_twoway_execute_with_collected_result(const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactories... shared_factories)
{
  // get a reference to avoid copies in agency::require
  executor_ref<E> exec_ref{exec};

  // wrap f in a functor that will collect f's result and call bulk_twoway_execute()
  return agency::require(exec_ref, bulk, twoway).bulk_twoway_execute(invoke_and_collect_result<Function>{f}, shape, result_factory, shared_factories...);
}


} // end detail
} // end agency

