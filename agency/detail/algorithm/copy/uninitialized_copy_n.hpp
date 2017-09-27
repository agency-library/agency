#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/algorithm/construct_n.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Iterator1, class Size, class Iterator2,
         __AGENCY_REQUIRES(is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value)>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy_n(ExecutionPolicy&& policy, Iterator1 first, Size n, Iterator2 result)
{
  return detail::construct_n(std::forward<ExecutionPolicy>(policy), result, n, first);
}


template<class Iterator1, class Size, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy_n(Iterator1 first, Size n, Iterator2 result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  sequenced_execution_policy seq;
  return detail::uninitialized_copy_n(seq, first, n, result);
}


} // end detail
} // end agency

