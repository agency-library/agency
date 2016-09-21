#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_continuation_executor.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T>
using is_simple_executor = agency::detail::disjunction<
  is_synchronous_executor<T>,
  is_asynchronous_executor<T>,
  is_continuation_executor<T>
>;


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool SimpleExecutor()
{
  return is_simple_executor<T>();
}


} // end new_executor_traits_detail
} // end detail
} // end agency
