#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_execution_depth_or.hpp>

namespace agency
{
namespace detail
{


template<class Executor, bool Enable = is_executor<Executor>::value>
struct executor_execution_depth_impl;

template<class Executor>
struct executor_execution_depth_impl<Executor,true>
  : executor_execution_depth_or<Executor>
{};


} // end detail


template<class Executor>
struct executor_execution_depth : detail::executor_execution_depth_impl<Executor> {};


} // end agency

