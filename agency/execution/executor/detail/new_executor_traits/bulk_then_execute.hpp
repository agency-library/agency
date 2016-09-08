#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_continuation_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_future.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/new_executor_traits/bulk_async_execute.hpp>
#include <future>


namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


// this case handles executors which .bulk_then_execute()
template<class E, class Function, class Future, class Factory1, class Factory2,
         __AGENCY_REQUIRES(BulkContinuationExecutor<E>())
        >
executor_future_t<
  E,
  result_of_t<Factory1()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factory1 result_factory, Factory2 shared_factory)
{
  return exec.bulk_then_execute(f, shape, predecessor, result_factory, shared_factory);
}


template<class Function, class SharedFuture,
         bool Enable = std::is_void<typename future_traits<SharedFuture>::value_type>::value
        >
struct bulk_then_execute_functor
{
  mutable Function f_;
  mutable SharedFuture fut_;

  using predecessor_type = typename future_traits<SharedFuture>::value_type;

  template<class Index, class... Args>
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
  Function f_;
  SharedFuture fut_;

  template<class Index, class... Args>
  auto operator()(const Index &idx, Args&... args) ->
    decltype(f_(idx, args...))
  {
    fut_.wait();

    return f_(idx, args...);
  }
};



// this case handles executors which have .bulk_async_execute() and may or may not have .bulk_execute()
template<class E, class Function, class Future, class Factory1, class Factory2,
         __AGENCY_REQUIRES(!BulkContinuationExecutor<E>() && BulkAsynchronousExecutor<E>())
        >
executor_future_t<
  E,
  result_of_t<Factory1()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factory1 result_factory, Factory2 shared_factory)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  using shared_predecessor_future_type = decltype(shared_predecessor_future);
  auto functor = bulk_then_execute_functor<Function,shared_predecessor_future_type>{f, shared_predecessor_future};

  return bulk_async_execute(exec, functor, shape, result_factory, shared_factory);
}


// this case handles executors which only have .bulk_execute()
template<class E, class Function, class Future, class Factory1, class Factory2,
         __AGENCY_REQUIRES(!BulkContinuationExecutor<E>() && !BulkAsynchronousExecutor<E>())
        >
executor_future_t<
  E,
  result_of_t<Factory1()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, Factory1 result_factory, Factory2 shared_factory)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  return std::async(std::launch::deferred, [=]() mutable
  {
    using shared_predecessor_future_type = decltype(shared_predecessor_future);
    auto functor = bulk_then_execute_functor<Function,shared_predecessor_future_type>{f, shared_predecessor_future};

    return bulk_execute(exec, functor, shape, result_factory, shared_factory);
  });
}


} // end new_executor_traits_detail
} // end detail
} // end agency

