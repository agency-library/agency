#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/adaptors/then_executor.hpp>


namespace agency
{


template<class E, class Function, class Future,
         __AGENCY_REQUIRES(detail::ContinuationExecutor<E>() or detail::BulkContinuationExecutor<E>())>
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_continuation_t<detail::decay_t<Function>,Future>
>
then_execute(E& exec, Function&& f, Future& predecessor)
{
  return detail::then_executor<E>(exec).then_execute(std::forward<Function>(f), predecessor);
}


} // end agency

