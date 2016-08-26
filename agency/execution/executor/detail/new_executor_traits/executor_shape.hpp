#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_shape_type_or.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class BulkExecutor, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_shape_impl
{
};

template<class BulkExecutor>
struct executor_shape_impl<BulkExecutor,true>
{
  using type = member_shape_type_or_t<BulkExecutor,std::size_t>;
};


template<class BulkExecutor>
struct executor_shape : executor_shape_impl<BulkExecutor> {};

template<class BulkExecutor>
using executor_shape_t = typename executor_shape<BulkExecutor>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency

