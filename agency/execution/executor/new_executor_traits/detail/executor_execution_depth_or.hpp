#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/new_executor_traits/detail/member_execution_category_or.hpp>

namespace agency
{
namespace detail
{


template<class T>
struct executor_execution_depth_or
  : agency::detail::execution_depth<
      member_execution_category_or_t<T,unsequenced_execution_tag>
    >
{};


} // end detail
} // end agency

