#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_execution_category_or.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T>
struct executor_execution_depth_or
  : agency::detail::execution_depth<
      member_execution_category_or_t<T,unsequenced_execution_tag>
    >
{};


} // end new_executor_traits_detail
} // end detail
} // end agency

