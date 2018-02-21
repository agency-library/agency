#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/then_execute_via_bulk_then_execute.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/then_execute_via_bulk_twoway_execute.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <utility>


namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class Executor, class Function, class Future,
         __AGENCY_REQUIRES(
           is_single_then_executor<Executor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
  then_execute(const Executor& ex, Function&& f, Future& fut)
{
  return ex.then_execute(std::forward<Function>(f), fut);
}

template<class Executor, class Function, class Future,
         __AGENCY_REQUIRES(
           !is_single_then_executor<Executor>::value and
           is_bulk_then_executor<Executor>::value
        )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
  then_execute(const Executor& ex, Function&& f, Future& fut)
{
  return detail::then_execute_via_bulk_then_execute(ex, std::forward<Function>(f), fut);
}

// XXX this is currently unimplemented
//template<class Executor, class Function, class Future,
//         __AGENCY_REQUIRES(
//           !is_single_then_executor<Executor>::value and
//           !is_bulk_then_executor<Executor>::value
//           is_single_twoway_executor<Executor>::value
//         )>
//__AGENCY_ANNOTATION
//executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
//  then_execute(const Executor& ex, Function&& f, Future& fut);

template<class Executor, class Function, class Future,
         __AGENCY_REQUIRES(
           !is_single_then_executor<Executor>::value and
           !is_bulk_then_executor<Executor>::value and
           !is_single_twoway_executor<Executor>::value and
           is_bulk_twoway_executor<Executor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
  then_execute(const Executor& ex, Function&& f, Future& fut)
{
  return detail::then_execute_via_bulk_twoway_execute(ex, std::forward<Function>(f), fut);
}

// XXX implement when Agency supports oneway executors
//template<class Executor, class Function, class T,
//         __EXECUTORS_REQUIRES(
//           !is_single_then_executor<Executor>::value
//           and is_single_oneway_executor<Executor>::value
//         )>
//__AGENCY_ANNOTATION
//auto then_execute(const Executor& ex, Function&& f, std::experimental::future<T>& fut);


} // end detail
} // end agency

