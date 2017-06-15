#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <cassert>
#include <cstdio>

namespace agency
{
namespace detail
{
namespace construct_n_detail
{


struct construct_n_functor
{
  __agency_exec_check_disable__
  template<class Agent, class RandomAccessIterator, class... RandomAccessIterators>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, RandomAccessIterator first, RandomAccessIterators... iters)
  {
    auto i = self.rank();

    ::new(static_cast<void*>(&first[i])) typename std::iterator_traits<RandomAccessIterator>::value_type(iters[i]...);
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
template<class ExecutionPolicy, class RandomAccessIterator, class Size, class... RandomAccessIterators,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value          
         ),
         __AGENCY_REQUIRES(
            !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
            iterators_are_random_access<RandomAccessIterator,RandomAccessIterators...>::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator construct_n(ExecutionPolicy&& policy, RandomAccessIterator first, Size n, RandomAccessIterators... iters)
{
  agency::bulk_invoke(policy(n), construct_n_detail::construct_n_functor(), first, iters...);

  return first + n;
}


// this overload is for cases where we must execute sequentially
// 1. ExecutionPolicy is sequenced OR
// 2. Iterators are not random access
template<class ExecutionPolicy, class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value          
         ),
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterators_are_random_access<Iterator,Iterators...>::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(ExecutionPolicy&&, Iterator first, Size n, Iterators... iters)
{
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  for(Size i = 0; i < n; ++i, ++first, construct_n_detail::swallow(++iters...))
  {
    new(&*first) value_type(*iters...);
  }

  return first;
}


// XXX we introduce an Allocator parameter here if we can figure out how to support it
//     internally, construct_n would call allocator_traits<Allocator>::construct(alloc, ...)
template<class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           // XXX we have no is_iterator, so just use the negation of the requirement used above
           !is_execution_policy<Iterator>::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(Iterator first, Size n, Iterators... iters)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  sequenced_execution_policy seq;
  return detail::construct_n(seq, first, n, iters...);
}


} // end detail
} // end agency

