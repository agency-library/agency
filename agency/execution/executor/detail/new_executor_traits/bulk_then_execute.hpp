#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_continuation_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>
#include <future>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// this case handles executors which have .bulk_then_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(BulkContinuationExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_then_execute(f, shape, predecessor, result_factory, shared_factories...);
}


template<class Function, class SharedFuture,
         bool Enable = std::is_void<typename future_traits<SharedFuture>::value_type>::value
        >
struct bulk_then_execute_functor
{
  mutable Function f_;
  mutable SharedFuture fut_;

  using predecessor_type = typename future_traits<SharedFuture>::value_type;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  ~bulk_then_execute_functor() = default;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bulk_then_execute_functor(Function f, const SharedFuture& fut)
    : f_(f), fut_(fut)
  {}

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bulk_then_execute_functor(const bulk_then_execute_functor&) = default;

  __agency_exec_check_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index &idx, Args&... args) const ->
    decltype(f_(idx, const_cast<predecessor_type&>(fut_.get()),args...))
  {
    predecessor_type& predecessor = const_cast<predecessor_type&>(fut_.get());

    return f_(idx, predecessor, args...);
  }
};


template<class Function, class SharedFuture>
struct bulk_then_execute_functor<Function,SharedFuture,true>
{
  mutable Function f_;
  mutable SharedFuture fut_;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  ~bulk_then_execute_functor() = default;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bulk_then_execute_functor(Function f, const SharedFuture& fut)
    : f_(f), fut_(fut)
  {}

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bulk_then_execute_functor(const bulk_then_execute_functor&) = default;

  __agency_exec_check_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index &idx, Args&... args) const ->
    decltype(f_(idx, args...))
  {
    fut_.wait();

    return f_(idx, args...);
  }
};



// this case handles executors which have .bulk_async_execute() and may or may not have .bulk_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(!BulkContinuationExecutor<E>() && BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  using shared_predecessor_future_type = decltype(shared_predecessor_future);
  auto functor = bulk_then_execute_functor<Function,shared_predecessor_future_type>{f, shared_predecessor_future};

  return bulk_async_execute(exec, functor, shape, result_factory, shared_factories...);
}


// this case handles executors which only have .bulk_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(!BulkContinuationExecutor<E>() && !BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  // XXX we should call async_execute(exec, ...) instead of std::async() here
  // XXX alternatively, we could call then_execute(exec, predecessor, ...) and not wait inside the function
  
  // XXX need to use a __host__ __device__ functor here instead of a lambda

  return std::async(std::launch::deferred, [=]() mutable
  {
    using shared_predecessor_future_type = decltype(shared_predecessor_future);
    auto functor = bulk_then_execute_functor<Function,shared_predecessor_future_type>{f, shared_predecessor_future};

    return bulk_execute(exec, functor, shape, result_factory, shared_factories...);
  });
}


} // end new_executor_traits_detail
} // end detail
} // end agency

