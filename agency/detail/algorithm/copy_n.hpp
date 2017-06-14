#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/iterator/iterator_traits.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{
namespace detail
{
namespace copy_n_detail
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
           iterator_is_random_access<RandomAccessIterator1>::value and
           iterator_is_random_access<RandomAccessIterator2>::value
         )>
__AGENCY_ANNOTATION
tuple<RandomAccessIterator1,RandomAccessIterator2> default_copy_n(ExecutionPolicy&& policy, RandomAccessIterator1 first, Size n, RandomAccessIterator2 result)
{
  agency::bulk_invoke(policy(n), copy_n_functor(), first, result);
  
  return detail::make_tuple(first + n, result + n);
}


template<class ExecutionPolicy, class InputIterator, class Size, class OutputIterator,
         __AGENCY_REQUIRES(
           policy_is_sequenced<decay_t<ExecutionPolicy>>::value or
           !iterator_is_random_access<InputIterator>::value or
           !iterator_is_random_access<OutputIterator>::value
         )>
__AGENCY_ANNOTATION
tuple<InputIterator,OutputIterator> default_copy_n(ExecutionPolicy&&, InputIterator first, Size n, OutputIterator result)
{
  // XXX we might wish to bulk_invoke a single agent and execute this loop inside

  for(Size i = 0; i < n; ++i, ++first, ++result)
  {
    *result = *first;
  }

  return detail::make_tuple(first, result);
}


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
      return copy_n_detail::default_copy_n(std::forward<ExecutionPolicy>(policy), first, n, result);
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
  constexpr copy_n_detail::copy_n_t copy_n{};
} // end namespace


} // end detail
} // end agency

