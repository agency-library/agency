#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/utility/invoke_functors.hpp>
#include <agency/execution/executor/customization_points/bulk_then_execute.hpp>
#include <agency/future.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class E, class Function, class Future, class ResultFactory, class... SharedFactories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(SharedFactories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<SharedFactories()>&...>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,result_of_t<ResultFactory()>>
  bulk_then_execute_with_collected_result(const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, SharedFactories... shared_factories)
{
  using predecessor_type = future_value_t<Future>;

  // wrap f in a functor that will collect f's result and call bulk_then_execute()
  return agency::bulk_then_execute(exec, invoke_and_collect_result<Function,predecessor_type>{f}, shape, predecessor, result_factory, shared_factories...);
}


} // end detail
} // end agency

