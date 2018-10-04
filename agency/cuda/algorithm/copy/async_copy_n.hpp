#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/algorithm/copy/async_copy_n.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <agency/cuda/future.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <iterator>

namespace agency
{
namespace cuda
{
namespace detail
{
namespace async_copy_n_detail
{


template<class Iterator1, class Iterator2>
using iterator_values_are_trivially_copyable = agency::detail::conjunction<
  std::is_same<typename std::iterator_traits<Iterator1>::value_type, typename std::iterator_traits<Iterator2>::value_type>,
  agency::detail::iterator_value_is_trivially_copyable<Iterator1>
>;


// this is the implementation of async_copy_n for contiguous, trivially copyable types
template<class ExecutionPolicy, class RandomAccessIterator1, class Size, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
           iterator_values_are_trivially_copyable<RandomAccessIterator1,RandomAccessIterator2>::value

           // XXX eliminate this requirement when future_cast() (which would be used to cast the return) is fully functional
           and std::is_same<execution_policy_future_t<ExecutionPolicy,void>, async_future<void>>::value
         )>
execution_policy_future_t<ExecutionPolicy,void> async_copy_n(ExecutionPolicy&&, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  // XXX we leak this stream if we throw an exception below
  cudaStream_t stream = experimental::make_dependent_stream(cuda::make_ready_async_future());

  using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
  const value_type* source = &*first;
  value_type* dest = &*result;

  // note our use of cudaMemcpyAsync avoids synchronizing the entire system, unlike cudaMemcpy
  // XXX we need to specify which device to use to perform the copy
  detail::throw_on_error(cudaMemcpyAsync(dest, source, n * sizeof(value_type), cudaMemcpyDefault, stream), "cuda::async_copy_n(): After cudaMemcpyAsync()");

  // make a future corresponding to the eventual completion of the copy
  async_future<void> future_result = experimental::make_async_future(stream);

  // destroy the stream
  detail::throw_on_error(cudaStreamDestroy(stream), "cuda::async_copy_n(): After cudaStreamDestroy()");

  return future_result;
}


// this is the implementation of async_copy_n for iterators which are not contiguous nor trivially copyable
template<class ExecutionPolicy, class RandomAccessIterator1, class Size, class RandomAccessIterator2,
         __AGENCY_REQUIRES(
           !iterator_values_are_trivially_copyable<RandomAccessIterator1,RandomAccessIterator2>::value

           // XXX eliminate this requirement when the complementary requirement used in the overload above is eliminated
           or !std::is_same<execution_policy_future_t<ExecutionPolicy,void>, async_future<void>>::value
         )>
execution_policy_future_t<ExecutionPolicy,void> async_copy_n(ExecutionPolicy&& policy, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  return agency::detail::default_async_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
}


} // end async_copy_n_detail
} // end detail


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
execution_policy_future_t<ExecutionPolicy,void> async_copy_n(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
{
  return detail::async_copy_n_detail::async_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
}


} // end cuda
} // end agency

