#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute_with_auto_result.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>


namespace agency
{
namespace detail
{
namespace blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_detail
{


template<class Function>
struct ignore_shared_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class... IgnoredArgs>
  __AGENCY_ANNOTATION
  result_of_t<Function(const Index&)> operator()(const Index& idx, IgnoredArgs&...) const
  {
    return agency::detail::invoke(f, idx);
  }
};


template<size_t>
using factory_returning_ignored_result = agency::detail::unit_factory;


template<size_t... Indices, class E, class Function>
__AGENCY_ANNOTATION
auto blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_impl(index_sequence<Indices...>,
                                                                                      const E& exec,
                                                                                      Function f,
                                                                                      executor_shape_t<E> shape) ->

  decltype(
    blocking_bulk_twoway_execute_with_auto_result(
      exec,
      ignore_shared_parameters_and_invoke<Function>{f},
      shape,
      factory_returning_ignored_result<Indices>()...
    )
  )
{
  return blocking_bulk_twoway_execute_with_auto_result(
    exec,                                             // the executor
    ignore_shared_parameters_and_invoke<Function>{f}, // the functor to execute
    shape,                                            // the number of agents to create
    factory_returning_ignored_result<Indices>()...    // pass a factory for each level of execution hierarchy. the results of these factories will be ignored
  );
}


} // end blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_detail


template<class E, class Function, __AGENCY_REQUIRES(is_executor<E>::value)>
__AGENCY_ANNOTATION
auto blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters(E& exec,
                                                                                 Function f,
                                                                                 executor_shape_t<E> shape) ->
  decltype(
    blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_detail::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_impl(
      detail::make_index_sequence<executor_execution_depth<E>::value>(),
      exec,
      f,
      shape
    )
  )
{
  namespace ns = blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_detail;

  return ns::blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters_impl(
    detail::make_index_sequence<executor_execution_depth<E>::value>(),
    exec,
    f,
    shape
  );
} // end blocking_bulk_twoway_execute_with_auto_result_and_without_shared_parameters()


} // end detail
} // end agency

