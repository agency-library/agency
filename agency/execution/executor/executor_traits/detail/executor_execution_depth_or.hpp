#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/execution/executor/properties/detail/bulk_guarantee_depth.hpp>


namespace agency
{
namespace detail
{



template<class T>
struct executor_execution_depth_or
  : bulk_guarantee_depth<
      decltype(bulk_guarantee_t::template static_query<T>())
    >
{};


} // end detail
} // end agency

