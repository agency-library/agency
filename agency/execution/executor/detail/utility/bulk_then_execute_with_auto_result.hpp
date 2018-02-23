#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_void_result.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_with_collected_result.hpp>
#include <agency/execution/executor/detail/utility/executor_bulk_result.hpp>
#include <agency/detail/factory.hpp>


namespace agency
{
namespace detail
{


// this is the case for when Function returns void
__agency_exec_check_disable__
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,void>
  bulk_then_execute_with_auto_result(const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  return bulk_then_execute_with_void_result(exec, f, shape, predecessor, factories...);
}


// this is the case for when Function returns non-void
// when Function does not return void, this function collects
// the results of each invocation into a container
// this container is returned through a future
template<class E, class Function, class Future, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories)),
         __AGENCY_REQUIRES(!std::is_void<result_of_continuation_t<Function, executor_index_t<E>, Future, result_of_t<Factories()>&...>>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E,
  executor_bulk_result_t<E,
    result_of_continuation_t<Function,executor_index_t<E>,Future,result_of_t<Factories()>&...>
  >
>
  bulk_then_execute_with_auto_result(const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factories... factories)
{
  // compute the type of f's result
  using result_type = result_of_continuation_t<Function,executor_index_t<E>,Future,result_of_t<Factories()>&...>;

  // compute the type of container that will store f's results
  using container_type = executor_bulk_result_t<E,result_type>;
  
  // create a factory that will construct this type of container for us
  auto result_factory = detail::make_construct<container_type>(shape);

  // lower onto bulk_sync_execute_with_collected_result() with this result_factory
  return detail::bulk_then_execute_with_collected_result(exec, f, shape, predecessor, result_factory, factories...);
}


} // end detail
} // end agency

