#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
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


template<class ExecutionPolicy, class RandomAccessIterator1, class Size, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
           !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
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
tuple<RandomAccessIterator1,RandomAccessIterator2> copy_n(ExecutionPolicy&& policy, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  agency::bulk_invoke(policy(n), copy_n_functor(), first, result);
  
  return detail::make_tuple(first + n, result + n);
}


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !std::is_convertible<
             typename std::iterator_traits<InputIterator>::iterator_category,
             std::random_access_iterator_tag
           >::value or
           !std::is_convertible<
             typename std::iterator_traits<OutputIterator>::iterator_category,
             std::random_access_iterator_tag
           >::value
         )>
__AGENCY_ANNOTATION
tuple<InputIterator,OutputIterator> copy_n(ExecutionPolicy&&, InputIterator first, Size n, OutputIterator result)
{
  // XXX we might wish to bulk_invoke a single agent and execute this loop inside

  for(Size i = 0; i < n; ++i, ++first, ++result)
  {
    *result = *first;
  }

  return detail::make_tuple(first, result);
}


template<class InputIterator, class Size, class OutputIterator>
__AGENCY_ANNOTATION
tuple<InputIterator,OutputIterator> copy_n(InputIterator first, Size n, OutputIterator result)
{
  return detail::copy_n(agency::sequenced_execution_policy(), first, n, result);
}


} // end detail
} // end agency

