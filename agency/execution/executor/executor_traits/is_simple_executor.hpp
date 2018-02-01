#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_then_executor.hpp>

namespace agency
{


// XXX nomerge
//     this seems superfluous - eliminate it
template<class T>
using is_simple_executor = agency::detail::disjunction<
  detail::is_twoway_executor<T>,
  detail::is_then_executor<T>
>;


namespace detail
{


// XXX nomerge
//     this seems superfluous - eliminate it
// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool SimpleExecutor()
{
  return is_simple_executor<T>();
}


} // end detail
} // end agency

