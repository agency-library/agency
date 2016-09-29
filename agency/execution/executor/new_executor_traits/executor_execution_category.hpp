#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/new_executor_traits/detail/member_execution_category_or.hpp>
#include <agency/execution/execution_categories.hpp>

namespace agency
{
namespace detail
{


template<class BulkExecutor, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_execution_category_impl
{
};

template<class BulkExecutor>
struct executor_execution_category_impl<BulkExecutor,true>
{
  using type = member_execution_category_or_t<BulkExecutor,unsequenced_execution_tag>;
};


} // end detail


template<class BulkExecutor>
struct executor_execution_category : detail::executor_execution_category_impl<BulkExecutor> {};

template<class BulkExecutor>
using executor_execution_category_t = typename executor_execution_category<BulkExecutor>::type;


} // end agency

