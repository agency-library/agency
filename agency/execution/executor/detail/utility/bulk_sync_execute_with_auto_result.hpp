#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_sync_execute_with_collected_result.hpp>
#include <agency/container/executor_container.hpp>
#include <agency/detail/factory.hpp>


namespace agency
{
namespace detail
{


// this is the case for when Function returns void
__agency_exec_check_disable__
template<class E, class Function, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(std::is_void<result_of_t<Function(executor_index_t<E>, result_of_t<Factories()>&...)>>::value)
        >
__AGENCY_ANNOTATION
void bulk_sync_execute_with_auto_result(E& exec, Function f, executor_shape_t<E> shape, Factories... factories)
{
  return detail::bulk_sync_execute_with_void_result(exec, f, shape, factories...);
}


// this is the case for when Function returns non-void
// in this case, this function collects
// the results of each invocation into a container
// this container is returned through a future
template<class E, class Function, class... Factories,
         __AGENCY_REQUIRES(BulkExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_t<Function(executor_index_t<E>, result_of_t<Factories()>&...)>>::value)
        >
__AGENCY_ANNOTATION
executor_container<E,
  result_of_t<Function(executor_index_t<E>,result_of_t<Factories()>&...)>
>
  bulk_sync_execute_with_auto_result(E& exec, Function f, executor_shape_t<E> shape, Factories... factories)
{
  // compute the type of f's result
  using result_type = result_of_t<Function(executor_index_t<E>,result_of_t<Factories()>&...)>;

  // compute the type of container that will store f's results
  using container_type = executor_container<E,result_type>;
  
  // create a factory that will construct this type of container for us
  auto result_factory = detail::make_construct<container_type>(shape);

  // lower onto bulk_sync_execute_with_collected_result() with this result_factory
  return detail::bulk_sync_execute_with_collected_result(exec, f, shape, result_factory, factories...);
}


} // end detail
} // end agency

