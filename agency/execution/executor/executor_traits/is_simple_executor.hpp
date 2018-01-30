#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/is_synchronous_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/is_continuation_executor.hpp>

namespace agency
{


template<class T>
using is_simple_executor = agency::detail::disjunction<
  is_synchronous_executor<T>,
  detail::is_twoway_executor<T>,
  is_continuation_executor<T>
>;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool SimpleExecutor()
{
  return is_simple_executor<T>();
}


} // end detail
} // end agency

