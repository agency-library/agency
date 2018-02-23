#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <agency/execution/executor/properties/bulk.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/require.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/integer_sequence.hpp>


namespace agency
{
namespace detail
{
namespace bulk_twoway_execute_without_shared_parameters_detail
{


template<class Function>
struct ignore_shared_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class Result, class... IgnoredArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Result& result, IgnoredArgs&...) const
  {
    agency::detail::invoke(f, idx, result);
  }
};


template<size_t>
using factory_returning_ignored_result = agency::detail::unit_factory;


template<size_t... Indices, class E, class Function, class ResultFactory>
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_twoway_execute_without_shared_parameters_impl(index_sequence<Indices...>,
                                                     const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory)
{
  bulk_twoway_execute_without_shared_parameters_detail::ignore_shared_parameters_and_invoke<Function> execute_me{f};

  // get a reference to exec to avoid copies in agency::require
  executor_ref<E> exec_ref{exec};

  return agency::require(exec_ref, bulk, twoway).bulk_twoway_execute(
    execute_me,                                     // the functor to execute
    shape,                                          // the number of agents to create
    result_factory,                                 // the factory to create the result
    factory_returning_ignored_result<Indices>()...  // pass a factory for each level of execution hierarchy. the results of these factories will be ignored
  );
}


} // end bulk_twoway_execute_without_shared_parameters_detail


template<class E, class Function, class ResultFactory,
         __AGENCY_REQUIRES(is_executor<E>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_twoway_execute_without_shared_parameters(const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory)
{
  return bulk_twoway_execute_without_shared_parameters_detail::bulk_twoway_execute_without_shared_parameters_impl(
    detail::make_index_sequence<executor_execution_depth<E>::value>(),
    exec,
    f,
    shape,
    result_factory
  );
}


} // end detail
} // end agency

