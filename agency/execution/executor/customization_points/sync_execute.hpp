#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/new_executor_traits.hpp>
#include <agency/execution/executor/customization_points/async_execute.hpp>
#include <agency/execution/executor/customization_points/bulk_execute.hpp>


namespace agency
{


// this case handles executors which have .sync_execute()
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(detail::SynchronousExecutor<E>())
        >
__AGENCY_ANNOTATION
detail::result_of_t<detail::decay_t<Function>()>
  sync_execute(E& exec, Function&& f)
{
  return exec.sync_execute(std::forward<Function>(f));
}


// this case handles executors which are SimpleExecutors but do not have .sync_execute()
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(!detail::SynchronousExecutor<E>()),
         __AGENCY_REQUIRES(detail::SimpleExecutor<E>())
        >
__AGENCY_ANNOTATION
detail::result_of_t<detail::decay_t<Function>()>
  sync_execute(E& exec, Function&& f)
{
  return agency::async_execute(exec, std::forward<Function>(f)).get();
}


namespace detail
{


template<class Function>
struct sync_execute_functor
{
  mutable Function f;

  template<class Index, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result, unit) const
  {
    result = invoke_and_return_unit_if_void_result(f);
  }
};


} // end detail


// this case handles executors which have no way to create single-agent synchrony
__agency_exec_check_disable__
template<class E, class Function,
         __AGENCY_REQUIRES(!detail::SimpleExecutor<E>()),
         __AGENCY_REQUIRES(detail::BulkExecutor<E>())>
__AGENCY_ANNOTATION
detail::result_of_t<detail::decay_t<Function>()>
  sync_execute(E& exec, Function f)
{
  using result_of_function = detail::result_of_t<Function()>;

  // if f returns void, then return a unit from bulk_sync_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    detail::unit,
    result_of_function
  >::type;

  // XXX should really move f into this functor, but it's not clear how to make move-only
  //     parameters to CUDA kernels
  auto execute_me = detail::sync_execute_functor<Function>{f};

  using shape_type = executor_shape_t<E>;

  // call bulk_async_execute() and cast to the expected result, which handles void result
  return static_cast<result_of_function>(agency::bulk_execute(exec,
    execute_me,                        // the functor to execute
    detail::shape_cast<shape_type>(1), // create only a single agent
    detail::construct<result_type>(),  // a factory for creating f's result
    detail::unit_factory()             // a factory for creating a unit shared parameter which execute_me will ignore
  ));
}

  
} // end agency

