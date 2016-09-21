#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/integer_sequence.hpp>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{
namespace bulk_async_execute_with_one_shared_parameter_detail
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
  bulk_async_execute_with_one_shared_parameter_impl(index_sequence<Indices...>,
                                                    E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactory shared_factory)
{
  bulk_async_execute_with_one_shared_parameter_detail::ignore_trailing_shared_parameters_and_invoke<Function> execute_me{f};

  return bulk_async_execute(exec,
    execute_me,                                    // the functor to execute
    shape,                                         // the number of agents to create
    result_factory,                                // the factory to create the result
    shared_factory,                                // the factory to create the shared parameter
    factory_returning_ignored_result<Indices>()... // pass a factory for each inner level of execution hierarchy. the results of these factories will be ignored
  );
}


} // end bulk_async_execute_with_one_shared_parameter_detail


template<class E, class Function, class ResultFactory, class SharedFactory,
         __AGENCY_REQUIRES(BulkExecutor<E>())
        >
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_async_execute_with_one_shared_parameter(E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, SharedFactory shared_factory)
{
  return bulk_async_execute_with_one_shared_parameter_detail::bulk_async_execute_with_one_shared_parameter_impl(
    detail::make_index_sequence<executor_execution_depth<E>::value - 1>(),
    exec,
    f,
    shape,
    result_factory,
    shared_factory
  );
}


} // end new_executor_traits_detail
} // end detail
} // end agency

