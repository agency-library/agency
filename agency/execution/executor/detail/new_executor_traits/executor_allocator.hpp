#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_allocator_or.hpp>
#include <allocator>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class BulkExecutor, class T, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_allocator_impl
{
};

template<class BulkExecutor, class T>
struct executor_allocator_impl<BulkExecutor,T,true>
{
  using type = member_allocator_or_t<BulkExecutor,T,std::allocator>;
};


template<class BulkExecutor, class T>
struct executor_allocator : executor_allocator_impl<BulkExecutor,T> {};

template<class BulkExecutor, class T>
using executor_allocator_t = typename executor_allocator<BulkExecutor,T>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency


