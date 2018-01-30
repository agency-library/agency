#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/is_simple_executor.hpp>
#include <agency/execution/executor/executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_twoway_executor.hpp>

namespace agency
{


template<class T>
using is_executor = agency::detail::disjunction<
  is_simple_executor<T>,
  is_bulk_executor<T>,
  detail::is_single_twoway_executor<T>,
  detail::is_bulk_then_executor<T>,
  detail::is_bulk_twoway_executor<T>,
  detail::is_then_executor<T>,
  detail::is_twoway_executor<T>
>;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool Executor()
{
  return is_executor<T>();
}


} // end detail
} // end agency

