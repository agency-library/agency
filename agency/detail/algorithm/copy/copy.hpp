#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/algorithm/copy/copy_n.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>

namespace agency
{
namespace detail
{


template<class ExecutionPolicy, class RandomAccessIterator1, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
           !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
           iterators_are_random_access<RandomAccessIterator1,RandomAccessIterator2>::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator2 copy(ExecutionPolicy&& policy, RandomAccessIterator1 first, RandomAccessIterator1 last, RandomAccessIterator2 result)
{
  auto iter_pair = detail::copy_n(std::forward<ExecutionPolicy>(policy), first, last - first, result);
  return agency::get<1>(iter_pair);
}


template<class ExecutionPolicy, class InputIterator, class OutputIterator,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterators_are_random_access<InputIterator,OutputIterator>::value
         )>
__AGENCY_ANNOTATION
OutputIterator copy(ExecutionPolicy&&, InputIterator first, InputIterator last, OutputIterator result)
{
  // XXX we might wish to bulk_invoke a single agent and execute this loop inside

  for(; first != last; ++first, ++result)
  {
    *result = *first;
  }

  return result;
}


template<class InputIterator, class OutputIterator>
__AGENCY_ANNOTATION
OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  agency::sequenced_execution_policy seq;
  return detail::copy(seq, first, last, result);
}


} // end detail
} // end agency

