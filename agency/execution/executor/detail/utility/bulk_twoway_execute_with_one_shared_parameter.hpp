#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <agency/execution/executor/properties/bulk.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/integer_sequence.hpp>


namespace agency
{
namespace detail
{
namespace bulk_twoway_execute_with_one_shared_parameter_detail
{


template<class Function>
struct ignore_trailing_shared_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class Result, class SharedArg, class... IgnoredArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Result& result, SharedArg& shared_arg, IgnoredArgs&...) const
  {
    agency::detail::invoke(f, idx, result, shared_arg);
  }
};


template<size_t>
using factory_returning_ignored_result = agency::detail::unit_factory;


template<size_t... Indices, class E, class Function, class ResultFactory, class SharedFactory>
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_twoway_execute_with_one_shared_parameter_impl(index_sequence<Indices...>,
                                                     const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactory shared_factory)
{
  bulk_twoway_execute_with_one_shared_parameter_detail::ignore_trailing_shared_parameters_and_invoke<Function> execute_me{f};

  // grab a reference to the executor to avoid copies in require
  executor_ref<E> exec_ref{exec};

  return agency::require(exec_ref, bulk, twoway).bulk_twoway_execute(
    execute_me,                                    // the functor to execute
    shape,                                         // the number of agents to create
    result_factory,                                // the factory to create the result
    shared_factory,                                // the factory to create the shared parameter
    factory_returning_ignored_result<Indices>()... // pass a factory for each inner level of execution hierarchy. the results of these factories will be ignored
  );
}


} // end bulk_twoway_execute_with_one_shared_parameter_detail


template<class E, class Function, class ResultFactory, class SharedFactory,
         __AGENCY_REQUIRES(is_executor<E>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_twoway_execute_with_one_shared_parameter(const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactory shared_factory)
{
  return bulk_twoway_execute_with_one_shared_parameter_detail::bulk_twoway_execute_with_one_shared_parameter_impl(
    detail::make_index_sequence<executor_execution_depth<E>::value - 1>(),
    exec,
    f,
    shape,
    result_factory,
    shared_factory
  );
}


} // end detail
} // end agency

