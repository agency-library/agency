#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/new_executor_traits/detail/member_allocator_or.hpp>
#include <memory>


namespace agency
{
namespace detail
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


} // end detail


template<class BulkExecutor, class T>
struct new_executor_allocator : detail::executor_allocator_impl<BulkExecutor,T> {};

template<class BulkExecutor, class T>
using new_executor_allocator_t = typename new_executor_allocator<BulkExecutor,T>::type;


} // end agency

