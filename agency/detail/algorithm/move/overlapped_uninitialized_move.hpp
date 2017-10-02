#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/algorithm/move/uninitialized_move.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>

namespace agency
{
namespace detail
{
namespace overlapped_uninitialized_move_detail
{


template<class Iterator1, class Iterator2>
__AGENCY_ANNOTATION
Iterator2 uninitialized_move_backward(Iterator1 first, Iterator1 last, Iterator2 result)
{
  using value_type = typename std::iterator_traits<Iterator2>::value_type;

  // yes, we preincrement
  // the ranges are open on the right, i.e. [first, last)
  while(first != last)
  {
    new(&*result) value_type(std::move(*first));
  }

  return result;
}


} // end overlapped_uninitialized_move_detail


template<class ExecutionPolicy, class Iterator,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
         ),
         __AGENCY_REQUIRES(
           !policy_is_sequenced<decay_t<ExecutionPolicy>>::value and
           iterator_is_random_access<Iterator>::value
         )>
__AGENCY_ANNOTATION
Iterator overlapped_uninitialized_move(ExecutionPolicy&& policy, Iterator first, Iterator last, Iterator result)
{
  if(first < last && first <= result && result < last)
  {
    // result lies in [first, last)
    // it's safe to use uninitialized_move_backward here
    overlapped_uninitialized_move_detail::uninitialized_move_backward(first, last, result + (last - first));
    result += (last - first);
  }
  else
  {
    // result + (last - first) lies in [first, last)
    // it's safe to use uninitialized_move here
    result = agency::detail::uninitialized_move(std::forward<ExecutionPolicy>(policy), first, last, result);
  } // end else

  return result;
}


template<class ExecutionPolicy, class Iterator,
         __AGENCY_REQUIRES(
           is_execution_policy<typename std::decay<ExecutionPolicy>::type>::value
         ),
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterator_is_random_access<Iterator>::value
         )>
__AGENCY_ANNOTATION
Iterator overlapped_uninitialized_move(ExecutionPolicy&&, Iterator first, Iterator last, Iterator result)
{
  if(first < last && first <= result && result < last)
  {
    // result lies in [first, last)
    // it's safe to use uninitialized_move_backward here
    overlapped_uninitialized_move_detail::uninitialized_move_backward(first, last, result + (last - first));
    result += (last - first);
  }
  else
  {
    // result + (last - first) lies in [first, last)
    // it's safe to use uninitialized_move here
    agency::sequenced_execution_policy seq;
    result = agency::detail::uninitialized_move(seq, first, last, result);
  } // end else

  return result;
}


template<class Iterator>
__AGENCY_ANNOTATION
Iterator overlapped_uninitialized_move(Iterator first, Iterator last, Iterator result)
{
  // pass this instead of agency::seq to work around the prohibition on
  // taking the address of a global constexpr object (i.e., agency::seq) from a CUDA __device__ function
  agency::sequenced_execution_policy seq;
  return detail::overlapped_uninitialized_move(seq, first, last, result);
}


} // end detail
} // end agency

