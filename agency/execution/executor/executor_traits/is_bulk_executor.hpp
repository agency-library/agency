#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>

namespace agency
{


// XXX nomerge
// XXX eliminate is_bulk_executor
template<class T>
using is_bulk_executor = agency::detail::disjunction<
  detail::is_bulk_twoway_executor<T>,
  detail::is_bulk_then_executor<T>
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

