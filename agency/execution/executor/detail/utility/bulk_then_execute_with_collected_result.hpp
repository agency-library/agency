#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_then_execute.hpp>
#include <agency/future.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace bulk_then_execute_with_collected_result_detail
{


// XXX move this somewhere common so that both bulk_async_execute_with_collected_result() & bulk_execute_with_collected_result() can share it
// this definition is used when there is a non-void predecessor parameter
template<class Function, class Predecessor>
struct invoke_and_collect_result
{
  mutable Function f;

  __agency_exec_check_disable__
  template<class Index, class Collection, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, Collection& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }
};


// this definition is used when there is no predecessor parameter
template<class Function>
struct invoke_and_collect_result<Function,void>
{
  mutable Function f;

  __agency_exec_check_disable__
  template<class Index, class Collection, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Collection& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, shared_parameters...);
  }
};


} // end bulk_then_execute_with_collected_result_detail


template<class E, class Function, class Future, class ResultFactory, class... SharedFactories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(SharedFactories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<SharedFactories()>&...>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,result_of_t<ResultFactory()>>
  bulk_then_execute_with_collected_result(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, SharedFactories... shared_factories)
{
  using namespace bulk_then_execute_with_collected_result_detail;

  using predecessor_type = future_value_t<Future>;

  // wrap f in a functor that will collect f's result and call bulk_then_execute()
  return agency::bulk_then_execute(exec, invoke_and_collect_result<Function,predecessor_type>{f}, shape, predecessor, result_factory, shared_factories...);
}


} // end detail
} // end agency

