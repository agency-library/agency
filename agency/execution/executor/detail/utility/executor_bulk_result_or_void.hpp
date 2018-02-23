#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/detail/utility/executor_bulk_result.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


template<class Executor, class T, bool Enable = is_executor<Executor>::value>
struct executor_bulk_result_or_void {};

template<class Executor, class T>
struct executor_bulk_result_or_void<Executor,T,true>
{
  using type = typename detail::lazy_conditional<
    std::is_void<T>::value,
    detail::identity<void>,
    executor_bulk_result<Executor,T>
  >::type;
};


template<class Executor, class T>
using executor_bulk_result_or_void_t = typename executor_bulk_result_or_void<Executor,T>::type;


} // end detail
} // end agency

