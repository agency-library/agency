#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_async.hpp>
#include <agency/async.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <agency/tuple.hpp>

namespace agency
{
namespace detail
{


struct async_copy_n_functor
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
           iterators_are_random_access<RandomAccessIterator1,RandomAccessIterator2>::value
         )>
__AGENCY_ANNOTATION
execution_policy_future_t<ExecutionPolicy,void> default_async_copy_n(ExecutionPolicy&& policy, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  return agency::bulk_async(policy(n), async_copy_n_functor(), first, result);
}


struct sequenced_async_copy_n_functor
{
  __agency_exec_check_disable__
  template<class InputIterator, class Size, class OutputIterator>
  __AGENCY_ANNOTATION
  void operator()(InputIterator first, Size n, OutputIterator result)
  {
    for(Size i = 0; i < n; ++i, ++first, ++result)
    {
      *result = *first;
    }
  }
};


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterators_are_random_access<InputIterator,OutputIterator>::value
         )>
__AGENCY_ANNOTATION
execution_policy_future_t<ExecutionPolicy,void> default_async_copy_n(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
{
  return agency::async(policy.executor(), sequenced_async_copy_n_functor(), first, n, result);
}


namespace async_copy_n_detail
{


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
struct has_async_copy_n_free_function_impl
{
  template<class... Args,
           class = decltype(
             async_copy_n(std::declval<Args>()...)
          )>
  static std::true_type test(int);

  template<class...>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy,InputIterator,Size,OutputIterator>(0));
};

// this type trait reports whether async_copy_n(policy, first, n, result) is well-formed
// when async_copy_n is called as a free function (i.e., via ADL)
template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
using has_async_copy_n_free_function = typename has_async_copy_n_free_function_impl<ExecutionPolicy,InputIterator,Size,OutputIterator>::type;


// this is the type of the async_copy_n customization point
class async_copy_n_t
{
  private:
    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
             __AGENCY_REQUIRES(has_async_copy_n_free_function<ExecutionPolicy,InputIterator,Size,OutputIterator>::value)>
    __AGENCY_ANNOTATION
    static execution_policy_future_t<ExecutionPolicy,void> impl(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
    {
      // call async_copy_n() via ADL
      return async_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
             __AGENCY_REQUIRES(!has_async_copy_n_free_function<ExecutionPolicy,InputIterator,Size,OutputIterator>::value)>
    __AGENCY_ANNOTATION
    static execution_policy_future_t<ExecutionPolicy,void> impl(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
    {
      // call default_async_copy_n()
      return agency::detail::default_async_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

  public:
    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
    __AGENCY_ANNOTATION
    execution_policy_future_t<ExecutionPolicy,void> operator()(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result) const
    {
      return impl(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

    template<class InputIterator, class Size, class OutputIterator>
    __AGENCY_ANNOTATION
    execution_policy_future_t<agency::sequenced_execution_policy,void> operator()(InputIterator first, Size n, OutputIterator result) const
    {
      return operator()(agency::sequenced_execution_policy(), first, n, result);
    }
};


} // end async_copy_n_detail


namespace
{

// async_copy_n customization point

#ifndef __CUDA_ARCH__
constexpr async_copy_n_detail::async_copy_n_t async_copy_n{};
#else
// __device__ functions cannot access global variables, so make copy_n a __device__ variable in __device__ code
const __device__ async_copy_n_detail::async_copy_n_t async_copy_n;
#endif

} // end namespace


} // end detail
} // end agency

