#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy/detail/simple_sequenced_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <cassert>
#include <cstdio>

namespace agency
{
namespace detail
{
namespace construct_n_detail
{


template<class Allocator>
struct construct_n_functor
{
  Allocator alloc;

  __agency_exec_check_disable__
  template<class Agent, class RandomAccessIterator, class... RandomAccessIterators>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, RandomAccessIterator first, RandomAccessIterators... iters) const
  {
    auto i = self.rank();

    // make a copy of alloc because allocator_traits::construct() requires a mutable allocator
    Allocator mutable_alloc = alloc;

    detail::allocator_traits<Allocator>::construct(mutable_alloc, &first[i], iters[i]...);
  }
};


template<class... Args>
__AGENCY_ANNOTATION
int swallow(Args&&...)
{
  return 0;
}


} // end construct_n_detail


// this overload is for cases where we need not execute sequentially:
// 1. ExecutionPolicy is not sequenced AND
// 2. Iterators are random access
template<class ExecutionPolicy, class Allocator, class RandomAccessIterator, class Size, class... RandomAccessIterators,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value          
         ),
         __AGENCY_REQUIRES(
            !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
            iterators_are_random_access<RandomAccessIterator,RandomAccessIterators...>::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator construct_n(ExecutionPolicy&& policy, Allocator& alloc, RandomAccessIterator first, Size n, RandomAccessIterators... iters)
{
  agency::bulk_invoke(policy(n), construct_n_detail::construct_n_functor<Allocator>{alloc}, first, iters...);

  return first + n;
}


// this overload is for cases where we must execute sequentially
// 1. ExecutionPolicy is sequenced OR
// 2. Iterators are not random access
__agency_exec_check_disable__
template<class ExecutionPolicy, class Allocator, class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value          
         ),
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterators_are_random_access<Iterator,Iterators...>::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(ExecutionPolicy&&, Allocator& alloc, Iterator first, Size n, Iterators... iters)
{
  for(Size i = 0; i < n; ++i, ++first, construct_n_detail::swallow(++iters...))
  {
    detail::allocator_traits<Allocator>::construct(alloc, &*first, *iters...);
  }

  return first;
}


template<class Allocator, class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           // XXX we have no is_allocator, so just use the negation of the requirement used above
           !is_execution_policy<Allocator>::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(Allocator& alloc, Iterator first, Size n, Iterators... iters)
{
  // use simple_sequenced_policy here to avoid circular dependencies
  // created by the use of sequenced_policy
  simple_sequenced_policy<> seq;
  return detail::construct_n(seq, alloc, first, n, iters...);
}


} // end detail
} // end agency

