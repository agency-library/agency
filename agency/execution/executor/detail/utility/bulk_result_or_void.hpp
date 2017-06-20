#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_bulk_executor.hpp>
#include <agency/container/executor_container.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


template<class Executor, class T, bool Enable = is_bulk_executor<Executor>::value>
struct bulk_result_or_void {};

template<class Executor, class T>
struct bulk_result_or_void<Executor,T,true>
{
  using type = typename detail::lazy_conditional<
    std::is_void<T>::value,
    detail::identity<void>,
    detail::identity<bulk_result<T,executor_shape_t<Executor>, executor_allocator_t<Executor,T>>>
  >::type;
};


template<class Executor, class T>
using bulk_result_or_void_t = typename bulk_result_or_void<Executor,T>::type;


} // end detail
} // end agency

