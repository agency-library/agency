#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_without_shared_parameters.hpp>
#include <agency/detail/shape_cast.hpp>


namespace agency
{


// this case handles executors which have .then_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future,
         __AGENCY_REQUIRES(detail::ContinuationExecutor<E>())>
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_continuation_t<detail::decay_t<Function>,Future>
>
then_execute(E& exec, Function&& f, Future& predecessor)
{
  return exec.then_execute(std::forward<Function>(f), predecessor);
}


namespace detail
{


template<class Function>
struct then_execute_functor
{
  mutable Function f;

  // this overload of operator() handles the case when there is a non-void predecessor future
  template<class Index, class Predecessor, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Predecessor& predecessor, Result& result) const
  {
    result = invoke_and_return_unit_if_void_result(f, predecessor);
  }

  // this overload of operator() handles the case when there is a void predecessor future
  template<class Index, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result) const
  {
    result = invoke_and_return_unit_if_void_result(f);
  }
};


} // end detail


// this case handles executors which have .bulk_then_execute() but not .then_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future,
         __AGENCY_REQUIRES(!detail::ContinuationExecutor<E>()),
         __AGENCY_REQUIRES(detail::BulkContinuationExecutor<E>())>
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_continuation_t<detail::decay_t<Function>,Future>
>
then_execute(E& exec, Function f, Future& predecessor)
{
  using result_of_function = detail::result_of_continuation_t<Function,Future>;

  // if f returns void, then return a unit from bulk_then_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    detail::unit,
    result_of_function
  >::type;

  // XXX should really move f into this functor, but it's not clear how to make move-only
  //     parameters to CUDA kernels
  auto execute_me = detail::then_execute_functor<Function>{f};

  using shape_type = executor_shape_t<E>;

  // call bulk_then_execute_without_shared_parameters() to get an intermediate future
  auto intermediate_future = detail::bulk_then_execute_without_shared_parameters(
    exec,                              // the executor
    execute_me,                        // the functor to execute
    detail::shape_cast<shape_type>(1), // create only a single agent
    predecessor,                       // the incoming argument to f
    detail::construct<result_type>()   // a factory for creating f's result
  );

  // cast the intermediate future into the right type of future for the result
  return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
}


// XXX introduce a case to handle executors which only have .async_execute() ?
// XXX introduce a worst case which uses predecessor.then() and ignores the executor entirely?

  
} // end agency

