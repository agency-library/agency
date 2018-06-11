#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/algorithm/copy/uninitialized_copy_n.hpp>
#include <agency/detail/iterator/move_iterator.hpp>
#include <utility>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Allocator, class Iterator1, class Size, class Iterator2,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
         )>
__AGENCY_ANNOTATION
Iterator2 uninitialized_move_n(ExecutionPolicy&& policy, Allocator& alloc, Iterator1 first, Size n, Iterator2 result)
{
  return detail::uninitialized_copy_n(std::forward<ExecutionPolicy>(policy), alloc, detail::make_move_iterator(first), n, result);
}


template<class Allocator, class Iterator1, class Size, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_move_n(Allocator& alloc, Iterator1 first, Size n, Iterator2 result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  sequenced_execution_policy seq;
  return detail::uninitialized_move_n(seq, alloc, first, n, result);
}


} // end detail
} // end agency

