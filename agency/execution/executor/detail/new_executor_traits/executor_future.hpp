#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/new_executor_traits/member_future_or.hpp>
#include <future>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class BulkExecutor, class T, bool Enable = is_bulk_executor<BulkExecutor>::value>
struct executor_future_impl
{
};

template<class BulkExecutor, class T>
struct executor_future_impl<BulkExecutor,T,true>
{
  using type = member_future_or_t<BulkExecutor,T,std::future>;
};


template<class BulkExecutor, class T>
struct executor_future : executor_future_impl<BulkExecutor,T> {};

template<class BulkExecutor, class T>
using executor_future_t = typename executor_future<BulkExecutor,T>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency


