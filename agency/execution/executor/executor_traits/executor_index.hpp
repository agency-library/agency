#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/member_index_type_or.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{


template<class Executor, bool Enable = is_executor<Executor>::value>
struct executor_index_impl
{
};

template<class Executor>
struct executor_index_impl<Executor,true>
{
  using type = member_index_type_or_t<Executor,executor_shape_t<Executor>>;
};


} // end detail


template<class Executor>
struct executor_index : detail::executor_index_impl<Executor> {};

template<class Executor>
using executor_index_t = typename executor_index<Executor>::type;


} // end agency

