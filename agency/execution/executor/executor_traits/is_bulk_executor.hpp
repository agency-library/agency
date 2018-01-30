#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/executor_traits/is_bulk_continuation_executor.hpp>

namespace agency
{


template<class T>
using is_bulk_executor = agency::detail::disjunction<
  is_bulk_asynchronous_executor<T>,
  is_bulk_continuation_executor<T>,
  detail::is_bulk_twoway_executor<T>
>;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool BulkExecutor()
{
  return is_bulk_executor<T>();
}


} // end detail
} // end agency

