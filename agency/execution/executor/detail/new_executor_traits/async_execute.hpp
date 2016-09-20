#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_continuation_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// this case handles executors which have .async_execute()
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(AsynchronousExecutor<E>())
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<decay_t<Function>()>
>
async_execute(E& exec, Function&& f)
{
  return exec.async_execute(std::forward<Function>(f));
}


// this case handles executors which have .then_execute() but not .async_execute()
// XXX not really clear if we should prefer .bulk_async_execute() to calling then_execute()
// XXX one advantage of prioritizing an implementation using .then_execute() over .bulk_async_execute() is
//     that no intermediate future is involved
// XXX also, there's no weirdness involving move-only functions which .bulk_async_execute() would have trouble with
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(!AsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(ContinuationExecutor<E>())
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<decay_t<Function>()>
>
async_execute(E& exec, Function&& f)
{
  using void_future_type = executor_future_t<E,void>;

  // XXX should really allow the executor to participate here
  void_future_type ready_predecessor = future_traits<void_future_type>::make_ready();

  return exec.then_execute(std::forward<Function>(f), ready_predecessor);
}


namespace async_execute_detail
{


template<class Function>
struct functor
{
  mutable Function f;

  template<class Index, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result, unit) const
  {
    result = invoke_and_return_unit_if_void_result(f);
  }
};


} // end async_execute_detail


// this case handles executors which have no way to create single-agent asynchrony
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(!AsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(!ContinuationExecutor<E>()),
         __AGENCY_REQUIRES(BulkExecutor<E>())>
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<decay_t<Function>()>
>
async_execute(E& exec, Function f)
{
  using result_of_function = result_of_t<Function()>;

  // if f returns void, then return a unit from bulk_async_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    unit,
    result_of_function
  >::type;

  // XXX should really move f into this functor, but it's not clear how to make move-only
  //     parameters to CUDA kernels
  auto execute_me = async_execute_detail::functor<Function>{f};

  using shape_type = executor_shape_t<E>;

  // call bulk_async_execute() to get an intermediate future
  auto intermediate_future = bulk_async_execute(exec,
    execute_me,                // the functor to execute
    shape_cast<shape_type>(1), // create only a single agent
    construct<result_type>(),  // a factory for creating f's result
    unit_factory()             // a factory for creating a unit shared parameter which execute_me will ignore
  );

  // cast the intermediate future into the right type of future for the result
  return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
}

  
} // end new_executor_traits_detail
} // end detail
} // end agency


