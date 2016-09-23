#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/new_executor_traits/is_simple_executor.hpp>
#include <agency/execution/executor/new_executor_traits/is_bulk_executor.hpp>

namespace agency
{


template<class T>
using new_is_executor = agency::detail::disjunction<
  is_simple_executor<T>,
  is_bulk_executor<T>
>;


namespace detail
{


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool Executor()
{
  return new_is_executor<T>();
}


} // end detail
} // end agency

