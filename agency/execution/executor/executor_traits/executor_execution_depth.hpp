#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_execution_depth_or.hpp>

namespace agency
{
namespace detail
{


template<class BulkExecutor, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_execution_depth_impl;

template<class BulkExecutor>
struct executor_execution_depth_impl<BulkExecutor,true>
  : executor_execution_depth_or<BulkExecutor>
{};


} // end detail


template<class BulkExecutor>
struct executor_execution_depth : detail::executor_execution_depth_impl<BulkExecutor> {};


} // end agency

