#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/twoway_execute_via_bulk_then_execute.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/twoway_execute_via_bulk_twoway_execute.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/twoway_execute_via_then_execute.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <utility>


namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class Executor, class Function,
         __AGENCY_REQUIRES(is_single_twoway_executor<Executor>::value)
        >
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<decay_t<Function>()>>
  twoway_execute(const Executor& ex, Function&& f)
{
  return ex.twoway_execute(std::forward<Function>(f));
}


// this case handles executors which have .then_execute() but not .twoway_execute()
// XXX not really clear if we should prefer .bulk_twoway_execute() to calling .then_execute()
// XXX one advantage of prioritizing an implementation using .then_execute() over .bulk_twoway_execute() is
//     that no intermediate future is involved
// XXX also, there's no weirdness involving move-only functions which .bulk_twoway_execute() would have trouble with
template<class Executor, class Function,
         __AGENCY_REQUIRES(
           !is_single_twoway_executor<Executor>::value and
           is_single_then_executor<Executor>::value 
         )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<decay_t<Function>()>>
  twoway_execute(const Executor& ex, Function&& f)
{
  return detail::twoway_execute_via_then_execute(ex, std::forward<Function>(f));
}


// this case handles executors which have .bulk_twoway_execute() but not .twoway_execute()
template<class Executor, class Function,
         __AGENCY_REQUIRES(
           !is_single_twoway_executor<Executor>::value and
           !is_single_then_executor<Executor>::value and
           is_bulk_twoway_executor<Executor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<decay_t<Function>()>>
  twoway_execute(const Executor& ex, Function&& f)
{
  return detail::twoway_execute_via_bulk_twoway_execute(ex, std::forward<Function>(f));
}


// this case handles executors which have .bulk_then_execute() but not .twoway_execute() or .bulk_twoway_execute()
template<class Executor, class Function,
         __AGENCY_REQUIRES(
           !is_single_twoway_executor<Executor>::value and
           !is_single_then_executor<Executor>::value and
           !is_bulk_twoway_executor<Executor>::value and
           is_bulk_then_executor<Executor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<Executor, result_of_t<decay_t<Function>()>>
  twoway_execute(const Executor& ex, Function&& f)
{
  return detail::twoway_execute_via_bulk_then_execute(ex, std::forward<Function>(f));
}


} // end detail
} // end agency

