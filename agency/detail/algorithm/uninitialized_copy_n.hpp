#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


struct uninitialized_copy_n_functor
{
  __agency_exec_check_disable__
  template<class Agent, class Allocator, class RandomAccessIterator1, class RandomAccessIterator2>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, Allocator alloc, RandomAccessIterator1 first, RandomAccessIterator2 result)
  {
    auto i = self.rank();

    RandomAccessIterator1 from = first + i;
    RandomAccessIterator2 to = result + i;
    
    allocator_traits<Allocator>::construct(alloc, &*to, *from);
  }
};


// this overload is for cases where we need not execute sequentially
template<class ExecutionPolicy, class Allocator, class RandomAccessIterator1, class Size, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
            !policy_is_sequenced<typename std::decay<ExecutionPolicy>::type>::value and
            std::is_convertible<
              typename std::iterator_traits<RandomAccessIterator1>::iterator_category,
              std::random_access_iterator_tag
            >::value and
            std::is_convertible<
              typename std::iterator_traits<RandomAccessIterator2>::iterator_category,
              std::random_access_iterator_tag
            >::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator2 uninitialized_copy_n(ExecutionPolicy&& policy, const Allocator& alloc, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  agency::bulk_invoke(policy(n), uninitialized_copy_n_functor(), alloc, first, result);

  return result + n;
}


// this overload is for cases where we must execute sequentially
template<class ExecutionPolicy, class Allocator, class Iterator1, class Size, class Iterator2,
         __AGENCY_REQUIRES(
           policy_is_sequenced<typename std::decay<ExecutionPolicy>::type>::value or
           !std::is_convertible<
             typename std::iterator_traits<Iterator1>::iterator_category,
             std::random_access_iterator_tag
           >::value or
           !std::is_convertible<
             typename std::iterator_traits<Iterator2>::iterator_category,
             std::random_access_iterator_tag
           >::value
         )>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy_n(ExecutionPolicy&&, Allocator& alloc, Iterator1 first, Size n, Iterator2 result)
{
  auto iters = agency::detail::allocator_traits<Allocator>::construct_n(alloc, result, n, first);
  return agency::detail::get<0>(iters);
}


template<class Allocator, class Iterator1, class Size, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_copy_n(Allocator& alloc, Iterator1 first, Size n, Iterator2 result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  sequenced_execution_policy seq;
  return detail::uninitialized_copy_n(seq, alloc, first, n, result);
}


} // end detail
} // end agency

