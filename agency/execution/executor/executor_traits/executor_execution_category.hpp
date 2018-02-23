#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/member_execution_category_or.hpp>
#include <agency/execution/execution_categories.hpp>

namespace agency
{
namespace detail
{


template<class Executor, bool Enable = is_executor<Executor>::value>
struct executor_execution_category_impl
{
};

template<class Executor>
struct executor_execution_category_impl<Executor,true>
{
  using type = member_execution_category_or_t<Executor,unsequenced_execution_tag>;
};


} // end detail


template<class Executor>
struct executor_execution_category : detail::executor_execution_category_impl<Executor> {};

template<class Executor>
using executor_execution_category_t = typename executor_execution_category<Executor>::type;


} // end agency

