#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_execution_category_or.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T>
struct executor_execution_depth_or
  : agency::detail::execution_depth<
      executor_execution_category_or_t<T>
    >
{};


} // end new_executor_traits_detail
} // end detail
} // end agency

