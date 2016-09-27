#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/customization_points/bulk_async_execute.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace bulk_async_execute_with_collected_result_detail
{


// XXX move this somewhere common so that both bulk_async_execute_with_collected_result() & bulk_execute_with_collected_result() can share it
template<class Function>
struct invoke_and_collect_result
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


} // end bulk_async_execute_with_collected_result_detail


template<class E, class Function, class ResultFactory, class... SharedFactories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(SharedFactories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_t<Function(executor_index_t<E>, result_of_t<SharedFactories()>&...)>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,result_of_t<ResultFactory()>>
  bulk_async_execute_with_collected_result(E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactories... shared_factories)
{
  using namespace bulk_async_execute_with_collected_result_detail;

  // wrap f in a functor that will collect f's result and call bulk_async_execute()
  return agency::bulk_async_execute(exec, invoke_and_collect_result<Function>{f}, shape, result_factory, shared_factories...);
}


} // end detail
} // end agency

