#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/member_future_or.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class Function>
using async_execute_t = decltype(std::declval<T>().async_execute(std::declval<Function>()));


template<class T, class Function>
using has_async_execute = is_detected_exact<
  member_future_or_t<
    T,
    result_of_t<Function()>,
    std::future
  >,
  async_execute_t, T, Function
>;


template<class T>
using is_asynchronous_executor = has_async_execute<T, std::function<void()>>;


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool AsynchronousExecutor()
{
  return is_asynchronous_executor<T>();
}


} // end new_executor_traits_detail
} // end detail
} // end agency

