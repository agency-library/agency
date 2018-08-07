#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/is_allocator.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy/detail/simple_sequenced_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>


namespace agency
{
namespace detail
{


struct destroy_functor
{
  __agency_exec_check_disable__
  template<class Agent, class Allocator, class RandomAccessIterator>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, Allocator alloc, RandomAccessIterator first)
  {
    auto i = self.rank();

    allocator_traits<Allocator>::destroy(alloc, &first[i]);
  }
};


// this overload is for cases where we need not execute sequentially:
// 1. ExecutionPolicy is not sequenced AND
// 2. Iterator is random access
template<class ExecutionPolicy, class Allocator, class RandomAccessIterator,
         __AGENCY_REQUIRES(
           detail::is_allocator<Allocator>::value
         ),
         __AGENCY_REQUIRES(
           !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
           iterator_is_random_access<RandomAccessIterator>::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator destroy(ExecutionPolicy&& policy, const Allocator& alloc, RandomAccessIterator first, RandomAccessIterator last)
{
  auto n = last - first;

  agency::bulk_invoke(policy(n), destroy_functor(), alloc, first);

  return first + n;
}


// this overload is for cases where we must execute sequentially
// 1. ExecutionPolicy is sequenced OR
// 2. Iterators are not random access
template<class ExecutionPolicy, class Allocator, class Iterator,
         __AGENCY_REQUIRES(
           detail::is_allocator<Allocator>::value
         ),
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterator_is_random_access<Iterator>::value
         )>
__AGENCY_ANNOTATION
Iterator destroy(ExecutionPolicy&&, Allocator& alloc, Iterator first, Iterator last)
{
  // XXX perhaps we should bulk_invoke a single agent and execute this loop in that agent
  for(; first != last; ++first)
  {
    agency::detail::allocator_traits<Allocator>::destroy(alloc, &*first);
  }

  return first;
}


template<class Allocator, class Iterator,
         __AGENCY_REQUIRES(
           detail::is_allocator<Allocator>::value
         )>
__AGENCY_ANNOTATION
Iterator destroy(Allocator& alloc, Iterator first, Iterator last)
{
  // use simple_sequenced_policy here to avoid circular dependencies
  // created by the use of sequenced_policy
  simple_sequenced_policy<> seq;
  return detail::destroy(seq, alloc, first, last);
}


} // end detail
} // end agency

