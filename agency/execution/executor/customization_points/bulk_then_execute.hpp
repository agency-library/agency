#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <future>


namespace agency
{


// this case handles executors which have .bulk_then_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(detail::BulkContinuationExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  return exec.bulk_then_execute(f, shape, predecessor, result_factory, shared_factories...);
}


namespace detail
{


template<class Function, class SharedFuture,
         bool Enable = std::is_void<detail::future_value_t<SharedFuture>>::value
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


} // end detail


// this case handles executors which have .bulk_async_execute() and may or may not have .bulk_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(!detail::BulkContinuationExecutor<E>() && detail::BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  using shared_predecessor_future_type = decltype(shared_predecessor_future);
  auto functor = detail::bulk_then_execute_functor<Function,shared_predecessor_future_type>{f, shared_predecessor_future};

  return exec.bulk_async_execute(functor, shape, result_factory, shared_factories...);
}


namespace detail
{


// this functor is used by the implementation of bulk_then_execute() below which calls .then() with a nested bulk_execute() inside
// this definition is for the general case, when the predecessor Future type is non-void
template<class Executor, class Function, class Predecessor, class ResultFactory, class... SharedFactories>
struct then_with_nested_bulk_execute_functor
{
  mutable Executor exec;
  mutable Function f;
  executor_shape_t<Executor> shape;
  mutable ResultFactory result_factory;
  mutable detail::tuple<SharedFactories...> shared_factories;

  // this functor is passed to bulk_execute() below
  // it has a reference to the predecessor future to use as a parameter to f
  struct functor_for_bulk_execute
  {
    mutable Function f;
    Predecessor& predecessor;

    template<class Index, class Result, class... SharedArgs>
    __AGENCY_ANNOTATION
    void operator()(const Index& idx, Result& result, SharedArgs&... shared_args) const
    {
      agency::detail::invoke(f, idx, predecessor, result, shared_args...);
    }
  };

  __agency_exec_check_disable__
  template<size_t... Indices>
  __AGENCY_ANNOTATION
  result_of_t<ResultFactory()> impl(detail::index_sequence<Indices...>, Predecessor& predecessor) const
  {
    functor_for_bulk_execute functor{f, predecessor};

    return exec.bulk_execute(functor, shape, result_factory, detail::get<Indices>(shared_factories)...);
  }

  __AGENCY_ANNOTATION
  result_of_t<ResultFactory()> operator()(Predecessor& predecessor) const
  {
    return impl(detail::make_index_sequence<sizeof...(SharedFactories)>(), predecessor);
  }
};


// this specialization is for the case when the predecessor Future type is void
template<class Executor, class Function, class ResultFactory, class... SharedFactories>
struct then_with_nested_bulk_execute_functor<Executor,Function,void,ResultFactory,SharedFactories...>
{
  mutable Executor exec;
  mutable Function f;
  executor_shape_t<Executor> shape;
  mutable ResultFactory result_factory;
  mutable detail::tuple<SharedFactories...> shared_factories;

  __agency_exec_check_disable__
  template<size_t... Indices>
  __AGENCY_ANNOTATION
  result_of_t<ResultFactory()> impl(detail::index_sequence<Indices...>) const
  {
    return exec.bulk_execute(f, shape, result_factory, detail::get<Indices>(shared_factories)...);
  }

  // the predecessor future is void, so operator() receives no parameter
  __AGENCY_ANNOTATION
  result_of_t<ResultFactory()> operator()() const
  {
    return impl(detail::make_index_sequence<sizeof...(SharedFactories)>());
  }
};


} // end detail


// this case handles executors which only have .bulk_execute()
__agency_exec_check_disable__
template<class E, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(!detail::BulkContinuationExecutor<E>() && !detail::BulkAsynchronousExecutor<E>()),
         __AGENCY_REQUIRES(new_executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<ResultFactory()>
>
bulk_then_execute(E& exec, Function f, executor_shape_t<E> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  using predecessor_type = detail::future_value_t<Future>;
  detail::then_with_nested_bulk_execute_functor<E,Function,predecessor_type,ResultFactory,Factories...> functor{exec,f,shape,result_factory,detail::make_tuple(shared_factories...)};

  auto intermediate_fut = future_traits<Future>::then(predecessor, std::move(functor));
  using result_type = detail::result_of_t<ResultFactory()>;

  // XXX we may wish to allow the executor to participate in this cast operation
  return future_traits<decltype(intermediate_fut)>::template cast<result_type>(intermediate_fut);
}


} // end agency

