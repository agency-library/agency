#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_shape_type_or.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{


template<class BulkExecutor, bool Enable = agency::detail::new_executor_traits_detail::is_bulk_executor<BulkExecutor>::value>
struct executor_shape_impl
{
};

template<class BulkExecutor>
struct executor_shape_impl<BulkExecutor,true>
{
  using type = agency::detail::new_executor_traits_detail::member_shape_type_or_t<BulkExecutor,std::size_t>;
};


} // end detail


template<class BulkExecutor>
struct new_executor_shape : detail::executor_shape_impl<BulkExecutor> {};

template<class BulkExecutor>
using new_executor_shape_t = typename new_executor_shape<BulkExecutor>::type;


} // end agency

