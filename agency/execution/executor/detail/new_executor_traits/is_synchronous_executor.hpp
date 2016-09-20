#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class Function>
using sync_execute_t = decltype(std::declval<T>().sync_execute(std::declval<Function>()));


template<class T, class Function>
using has_sync_execute = is_detected_exact<result_of_t<Function()>, sync_execute_t, T, Function>;


template<class T>
using is_synchronous_executor = has_sync_execute<T, std::function<void()>>;


// a fake Concept to use with __AGENCY_REQUIRES
template<class T>
constexpr bool SynchronousExecutor()
{
  return is_synchronous_executor<T>();
}


} // end new_executor_traits_detail
} // end detail
} // end agency

