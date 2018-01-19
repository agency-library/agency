#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/default_shape.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/member_shape_type_or.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{


template<class Executor, bool Enable = is_executor<Executor>::value>
struct executor_shape_impl
{
};

template<class Executor>
struct executor_shape_impl<Executor,true>
{
  using type = member_shape_type_or_t<Executor,default_shape_t<1>>;
};


} // end detail


template<class Executor>
struct executor_shape : detail::executor_shape_impl<Executor> {};

template<class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


} // end agency

