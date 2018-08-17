#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/bulk_invoke.hpp>
#include <agency/execution/execution_policy/execution_policy_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>


namespace agency
{
namespace detail
{
namespace construct_array_detail
{


template<class Allocator>
struct construct_array_functor
{
  // mutable because allocator_traits::construct() requires a mutable allocator
  mutable Allocator alloc_;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  construct_array_functor(const Allocator& alloc)
    : alloc_(alloc)
  {}

  __AGENCY_ANNOTATION
  construct_array_functor(const construct_array_functor& other)
    : construct_array_functor(other.alloc_)
  {}

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  ~construct_array_functor() {}

  __agency_exec_check_disable__
  template<class Agent, class ArrayView, class... ArrayViews>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, ArrayView array, ArrayViews... arrays) const
  {
    auto idx = self.index();

    detail::allocator_traits<Allocator>::construct(alloc_, &array[idx], arrays[idx]...);
  }
};


template<class T, class Index>
struct is_indexable
{
  private:
    template<class U,
             class = decltype(std::declval<U>()[std::declval<Index>()])
            >
    static constexpr bool test(int) { return true; }

    template<class>
    static constexpr bool test(...) { return false; }

  public:
    static constexpr bool value = test<T>(0);
};


template<class ExecutionPolicy, class ArrayView, class... ArrayViews>
using construct_array_requirements = detail::conjunction<
  is_indexable<ArrayView, execution_policy_index_t<ExecutionPolicy>>,
  is_indexable<ArrayViews, execution_policy_index_t<ExecutionPolicy>>...
>;


template<class Alloc, class... Args>
struct has_construct_array_member
{
  private:
    template<class A,
             class = decltype(std::declval<A>().construct_array(std::declval<Args>()...))
            >
    static constexpr bool test(int) { return true; }

    template<class>
    static constexpr bool test(...) { return false; }

  public:
    static constexpr bool value = test<Alloc>(0);
};


} // end construct_array_detail


__agency_exec_check_disable__
template<class Allocator, class ExecutionPolicy, class ArrayView, class... ArrayViews,
         __AGENCY_REQUIRES(
           is_execution_policy<decay_t<ExecutionPolicy>>::value
         ),
         __AGENCY_REQUIRES(
           construct_array_detail::has_construct_array_member<Allocator, ExecutionPolicy&&, ArrayView, ArrayViews...>::value
         ),
         __AGENCY_REQUIRES(
           construct_array_detail::construct_array_requirements<decay_t<ExecutionPolicy>, ArrayView, ArrayViews...>::value
         )>
__AGENCY_ANNOTATION
void construct_array(Allocator& alloc, ExecutionPolicy&& policy, ArrayView array, ArrayViews... arrays)
{
  // call the allocator's member function
  alloc.construct_array(std::forward<ExecutionPolicy>(policy), array, arrays...);
}


__agency_exec_check_disable__
template<class Allocator, class ExecutionPolicy, class ArrayView, class... ArrayViews,
         __AGENCY_REQUIRES(
           is_execution_policy<decay_t<ExecutionPolicy>>::value
         ),
         __AGENCY_REQUIRES(
           !construct_array_detail::has_construct_array_member<Allocator, ExecutionPolicy&&, ArrayView, ArrayViews...>::value
         ),
         __AGENCY_REQUIRES(
           construct_array_detail::construct_array_requirements<decay_t<ExecutionPolicy>, ArrayView, ArrayViews...>::value
         )>
__AGENCY_ANNOTATION
void construct_array(Allocator& alloc, ExecutionPolicy&& policy, ArrayView array, ArrayViews... arrays)
{
  // generic implementation via bulk_invoke
  agency::bulk_invoke(
    policy(array.shape()),
    construct_array_detail::construct_array_functor<Allocator>{alloc},
    array,
    arrays...
  );
}


__agency_exec_check_disable__
template<class Allocator, class ArrayView, class... ArrayViews,
         __AGENCY_REQUIRES(
           !is_execution_policy<ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           construct_array_detail::has_construct_array_member<Allocator, ArrayView, ArrayViews...>::value
         )>
__AGENCY_ANNOTATION
void construct_array(Allocator& alloc, ArrayView array, ArrayViews... arrays)
{
  // call the allocator's member function
  alloc.construct_array(array, arrays...);
}


__agency_exec_check_disable__
template<class Allocator, class ArrayView, class... ArrayViews,
         __AGENCY_REQUIRES(
           !is_execution_policy<ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           !construct_array_detail::has_construct_array_member<Allocator, ArrayView, ArrayViews...>::value
         )>
__AGENCY_ANNOTATION
void construct_array(Allocator& alloc, ArrayView array, ArrayViews... arrays)
{
  // call construct() in a loop
  for(size_t i = 0; i < array.size(); ++i)
  {
    agency::detail::allocator_traits<Allocator>::construct(alloc, &array.begin()[i], arrays.begin()[i]...);
  }
}


} // end detail
} // end agency

