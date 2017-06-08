#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/is_allocator.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <iterator>

namespace agency
{
namespace detail
{


struct construct_n_functor
{
  __agency_exec_check_disable__
  template<class Agent, class Allocator, class RandomAccessIterator, class... RandomAccessIterators>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, Allocator alloc, RandomAccessIterator first, RandomAccessIterators... iters)
  {
    auto i = self.rank();

    allocator_traits<Allocator>::construct(alloc, &first[i], iters[i]...);
  }
};


// this overload is for cases where we need not execute sequentially:
// 1. ExecutionPolicy is not sequenced AND
// 2. Iterators are random access
template<class ExecutionPolicy, class Allocator, class RandomAccessIterator, class Size, class... RandomAccessIterators,
         __AGENCY_REQUIRES(
            !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
            conjunction<
              std::is_convertible<
                typename std::iterator_traits<RandomAccessIterator>::iterator_category,
                std::random_access_iterator_tag
              >,
              std::is_convertible<
                typename std::iterator_traits<RandomAccessIterators>::iterator_category,
                std::random_access_iterator_tag
              >...
            >::value
         )>
__AGENCY_ANNOTATION
RandomAccessIterator construct_n(ExecutionPolicy&& policy, const Allocator& alloc, RandomAccessIterator first, Size n, RandomAccessIterators... iters)
{
  agency::bulk_invoke(policy(n), construct_n_functor(), alloc, first, iters...);

  return first + n;
}


// this overload is for cases where we must execute sequentially
// 1. ExecutionPolicy is sequenced OR
// 2. Iterators are not random access
template<class ExecutionPolicy, class Allocator, class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !conjunction<
             std::is_convertible<
               typename std::iterator_traits<Iterator>::iterator_category,
               std::random_access_iterator_tag
             >,
             std::is_convertible<
               typename std::iterator_traits<Iterators>::iterator_category,
               std::random_access_iterator_tag
             >...
           >::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(ExecutionPolicy&&, Allocator& alloc, Iterator first, Size n, Iterators... iters)
{
  auto iter_tuple = agency::detail::allocator_traits<Allocator>::construct_n(alloc, first, n, iters...);
  return agency::detail::get<0>(iter_tuple);
}


template<class Allocator, class Iterator, class Size, class... Iterators,
         __AGENCY_REQUIRES(
           detail::is_allocator<Allocator>::value
         )>
__AGENCY_ANNOTATION
Iterator construct_n(Allocator& alloc, Iterator first, Size n, Iterators... iters)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  sequenced_execution_policy seq;
  return detail::construct_n(seq, alloc, first, n, iters...);
}


} // end detail
} // end agency


