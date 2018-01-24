#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/container/bulk_result.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


template<class Executor, class T, bool Enable = is_executor<Executor>::value>
struct executor_bulk_result {};

template<class Executor, class T>
struct executor_bulk_result<Executor,T,true>
{
  using type = bulk_result<
    T,
    executor_shape_t<Executor>,
    executor_allocator_t<Executor,T>
  >;
};


template<class Executor, class T>
using executor_bulk_result_t = typename executor_bulk_result<Executor,T>::type;


} // end detail
} // end agency


