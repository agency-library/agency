#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_synchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_asynchronous_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_continuation_executor.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{

template<class T>
using is_bulk_executor = agency::detail::disjunction<
  is_bulk_synchronous_executor<T>,
  is_bulk_asynchronous_executor<T>,
  is_bulk_continuation_executor<T>
>;


} // end new_executor_traits_detail
} // end detail
} // end agency

