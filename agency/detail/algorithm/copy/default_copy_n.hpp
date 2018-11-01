#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/functional/invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <agency/tuple.hpp>

namespace agency
{
namespace detail
{
namespace default_copy_n_detail
{


struct copy_n_functor
{
  __agency_exec_check_disable__
  template<class Agent, class RandomAccessIterator1, class RandomAccessIterator2>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, RandomAccessIterator1 first, RandomAccessIterator2 result)
  {
    auto i = self.rank();

    result[i] = first[i];
  }
};


struct sequenced_copy_n_functor
{
  __agency_exec_check_disable__
  template<class InputIterator, class Size, class OutputIterator>
  __AGENCY_ANNOTATION
  tuple<InputIterator,OutputIterator> operator()(InputIterator first, Size n, OutputIterator result)
  {
    for(Size i = 0; i < n; ++i, ++first, ++result)
    {
      *result = *first;
    }

    return agency::make_tuple(first, result);
  }
};


} // end default_copy_n_detail


template<class ExecutionPolicy, class RandomAccessIterator1, class Size, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
           !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
           iterators_are_random_access<RandomAccessIterator1,RandomAccessIterator2>::value
         )>
__AGENCY_ANNOTATION
tuple<RandomAccessIterator1,RandomAccessIterator2> default_copy_n(ExecutionPolicy&& policy, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  agency::bulk_invoke(policy(n), default_copy_n_detail::copy_n_functor(), first, result);
  
  return agency::make_tuple(first + n, result + n);
}


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterators_are_random_access<InputIterator,OutputIterator>::value
         )>
__AGENCY_ANNOTATION
tuple<InputIterator,OutputIterator> default_copy_n(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
{
  return agency::invoke(policy.executor(), default_copy_n_detail::sequenced_copy_n_functor(), first, n, result);
}


} // end detail
} // end agency

