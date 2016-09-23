#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/execution/executor/detail/customization_points/bulk_async_execute_with_one_shared_parameter.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>


namespace agency
{
namespace detail
{
namespace executor_customization_points_detail
{


// this case handles executors which have .async_execute()
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(AsynchronousExecutor<E>())
        >
__AGENCY_ANNOTATION
new_executor_future_t<
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
new_executor_future_t<
  E,
  result_of_t<decay_t<Function>()>
>
async_execute(E& exec, Function&& f)
{
  using void_future_type = new_executor_future_t<E,void>;

  // XXX should really allow the executor to participate here
  void_future_type ready_predecessor = future_traits<void_future_type>::make_ready();

  return exec.then_execute(std::forward<Function>(f), ready_predecessor);
}


namespace async_execute_detail
{


struct functor
{
  template<class Index, class Result, class SharedFunction>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result, SharedFunction& shared_function) const
  {
    result = invoke_and_return_unit_if_void_result(shared_function);
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
new_executor_future_t<
  E,
  result_of_t<decay_t<Function>()>
>
async_execute(E& exec, Function&& f)
{
  using result_of_function = result_of_t<Function()>;

  // if f returns void, then return a unit from bulk_async_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    unit,
    result_of_function
  >::type;

  using shape_type = new_executor_shape_t<E>;

  auto intermediate_future = bulk_async_execute_with_one_shared_parameter(
    exec,                                          // the executor
    async_execute_detail::functor(),               // the functor to execute
    shape_cast<shape_type>(1),                     // create only a single agent
    construct<result_type>(),                      // a factory for creating f's result
    make_moving_factory(std::forward<Function>(f)) // a factory to present f as the one shared parameter
  );

  // cast the intermediate future into the right type of future for the result
  return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
}

  
} // end executor_customization_points_detail
} // end detail
} // end agency


