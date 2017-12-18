#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/algorithm/copy/default_copy_n.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>


namespace agency
{
namespace detail
{
namespace copy_n_detail
{


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
struct has_copy_n_free_function_impl
{
  template<class... Args,
           class = decltype(
             copy_n(std::declval<Args>()...)
          )>
  static std::true_type test(int);

  template<class...>
  static std::false_type test(...);

  using type = decltype(test<ExecutionPolicy,InputIterator,Size,OutputIterator>(0));
};

// this type trait reports whether copy_n(policy, first, n, result) is well-formed
// when copy_n is called as a free function (i.e., via ADL)
template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
using has_copy_n_free_function = typename has_copy_n_free_function_impl<ExecutionPolicy,InputIterator,Size,OutputIterator>::type;


// this is the type of the copy_n customization point
class copy_n_t
{
  private:
    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
             __AGENCY_REQUIRES(has_copy_n_free_function<ExecutionPolicy,InputIterator,Size,OutputIterator>::value)>
    __AGENCY_ANNOTATION
    static tuple<InputIterator,OutputIterator> impl(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
    {
      // call copy_n() via ADL
      return copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
             __AGENCY_REQUIRES(!has_copy_n_free_function<ExecutionPolicy,InputIterator,Size,OutputIterator>::value)>
    __AGENCY_ANNOTATION
    static tuple<InputIterator,OutputIterator> impl(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result)
    {
      // call default_copy_n()
      return agency::detail::default_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

  public:
    template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator>
    __AGENCY_ANNOTATION
    tuple<InputIterator,OutputIterator> operator()(ExecutionPolicy&& policy, InputIterator first, Size n, OutputIterator result) const
    {
      return impl(std::forward<ExecutionPolicy>(policy), first, n, result);
    }

    template<class InputIterator, class Size, class OutputIterator>
    __AGENCY_ANNOTATION
    tuple<InputIterator,OutputIterator> operator()(InputIterator first, Size n, OutputIterator result) const
    {
      return operator()(agency::sequenced_execution_policy(), first, n, result);
    }
};


} // end copy_n_detail


namespace
{

// copy_n customization point

#ifndef __CUDA_ARCH__
constexpr copy_n_detail::copy_n_t copy_n{};
#else
// __device__ functions cannot access global variables, so make copy_n a __device__ variable in __device__ code
const __device__ copy_n_detail::copy_n_t copy_n;
#endif

} // end namespace


} // end detail
} // end agency

