#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_execution_depth_or.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class BulkExecutor, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_execution_depth_impl;

template<class BulkExecutor>
struct executor_execution_depth_impl<BulkExecutor,true>
  : executor_execution_depth_or<BulkExecutor>
{};


template<class BulkExecutor>
struct executor_execution_depth : executor_execution_depth_impl<BulkExecutor> {};


} // end new_executor_traits_detail
} // end detail
} // end agency

