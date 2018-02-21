#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/execution_functions/bulk_then_execute.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{
namespace bulk_then_execute_without_shared_parameters_detail
{


// in general, the predecessor future's type is non-void
template<class Function, class Predecessor>
struct ignore_shared_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class Result, class... IgnoredArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, Result& result, IgnoredArgs&...) const
  {
    agency::detail::invoke(f, idx, predecessor, result);
  }
};


// this specialization handles the void predecessor future case
template<class Function>
struct ignore_shared_parameters_and_invoke<Function,void>
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


template<size_t... Indices, class E, class Function, class Future, class ResultFactory>
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_then_execute_without_shared_parameters_impl(index_sequence<Indices...>,
                                                   const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory)
{
  using predecessor_type = future_result_t<Future>;
  bulk_then_execute_without_shared_parameters_detail::ignore_shared_parameters_and_invoke<Function,predecessor_type> execute_me{f};

  return detail::bulk_then_execute(exec,
    execute_me,                                     // the functor to execute
    shape,                                          // the number of agents to create
    predecessor,                                    // the predecessor future
    result_factory,                                 // the factory to create the result
    factory_returning_ignored_result<Indices>()...  // pass a factory for each level of execution hierarchy. the results of these factories will be ignored
  );
}


} // end bulk_then_execute_without_shared_parameters_detail


template<class E, class Function, class Future, class ResultFactory,
         __AGENCY_REQUIRES(is_executor<E>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<E, result_of_t<ResultFactory()>>
  bulk_then_execute_without_shared_parameters(const E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory)
{
  return bulk_then_execute_without_shared_parameters_detail::bulk_then_execute_without_shared_parameters_impl(
    detail::make_index_sequence<executor_execution_depth<E>::value>(),
    exec,
    f,
    shape,
    predecessor,
    result_factory
  );
}


} // end detail
} // end agency

