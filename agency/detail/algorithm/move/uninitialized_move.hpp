#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/algorithm/copy/uninitialized_copy.hpp>
#include <agency/detail/iterator/distance.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class Allocator, class ForwardIterator, class OutputIterator,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
         )>
__AGENCY_ANNOTATION
OutputIterator uninitialized_move(ExecutionPolicy&& policy, Allocator& alloc, ForwardIterator first, ForwardIterator last, OutputIterator result)
{
  return detail::uninitialized_copy(std::forward<ExecutionPolicy>(policy), alloc, detail::make_move_iterator(first), detail::make_move_iterator(last), result);
}


template<class Allocator, class ForwardIterator, class OutputIterator>
__AGENCY_ANNOTATION
OutputIterator uninitialized_move(Allocator& alloc, ForwardIterator first, ForwardIterator last, OutputIterator result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  agency::sequenced_execution_policy seq;
  return agency::detail::uninitialized_move(seq, alloc, first, last, result);
}


} // end detail
} // end agency

