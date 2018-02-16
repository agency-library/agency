#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_twoway_executor.hpp>

namespace agency
{


template<class T>
using is_executor = agency::detail::disjunction<
  detail::is_bulk_then_executor<T>,
  detail::is_bulk_twoway_executor<T>,
  detail::is_single_then_executor<T>,
  detail::is_single_twoway_executor<T>
>;


} // end agency

