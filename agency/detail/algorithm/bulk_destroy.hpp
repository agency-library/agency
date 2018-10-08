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
namespace bulk_destroy_detail
{


template<class Allocator>
struct bulk_destroy_functor
{
  // mutable because allocator_traits::destroy() requires a mutable allocator
  mutable Allocator alloc_;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  bulk_destroy_functor(const Allocator& alloc)
    : alloc_(alloc)
  {}

  __AGENCY_ANNOTATION
  bulk_destroy_functor(const bulk_destroy_functor& other)
    : bulk_destroy_functor(other.alloc_)
  {}

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  ~bulk_destroy_functor() {}

  __agency_exec_check_disable__
  template<class Agent, class ArrayView>
  __AGENCY_ANNOTATION
  void operator()(Agent& self, ArrayView array) const
  {
    auto idx = self.index();

    detail::allocator_traits<Allocator>::destroy(alloc_, &array[idx]);
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


template<class ExecutionPolicy, class ArrayView>
using bulk_destroy_requirements = detail::conjunction<
  is_indexable<ArrayView, execution_policy_index_t<ExecutionPolicy>>
>;


template<class Alloc, class... Args>
struct has_bulk_destroy_member
{
  private:
    template<class A,
             class = decltype(std::declval<A>().bulk_destroy(std::declval<Args>()...))
            >
    static constexpr bool test(int) { return true; }

    template<class>
    static constexpr bool test(...) { return false; }

  public:
    static constexpr bool value = test<Alloc>(0);
};


} // end bulk_destroy_detail


__agency_exec_check_disable__
template<class Allocator, class ExecutionPolicy, class ArrayView,
         __AGENCY_REQUIRES(
           is_execution_policy<decay_t<ExecutionPolicy>>::value
         ),
         __AGENCY_REQUIRES(
           bulk_destroy_detail::has_bulk_destroy_member<Allocator, ExecutionPolicy&&, ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           bulk_destroy_detail::bulk_destroy_requirements<decay_t<ExecutionPolicy>, ArrayView>::value
         )>
__AGENCY_ANNOTATION
void bulk_destroy(Allocator& alloc, ExecutionPolicy&& policy, ArrayView array)
{
  // call the allocator's member function
  alloc.bulk_destroy(std::forward<ExecutionPolicy>(policy), array);
}


__agency_exec_check_disable__
template<class Allocator, class ExecutionPolicy, class ArrayView,
         __AGENCY_REQUIRES(
           is_execution_policy<decay_t<ExecutionPolicy>>::value
         ),
         __AGENCY_REQUIRES(
           !bulk_destroy_detail::has_bulk_destroy_member<Allocator, ExecutionPolicy&&, ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           bulk_destroy_detail::bulk_destroy_requirements<decay_t<ExecutionPolicy>, ArrayView>::value
         )>
__AGENCY_ANNOTATION
void bulk_destroy(Allocator& alloc, ExecutionPolicy&& policy, ArrayView array)
{
  // generic implementation via bulk_invoke
  agency::bulk_invoke(
    policy(array.shape()),
    bulk_destroy_detail::bulk_destroy_functor<Allocator>{alloc},
    array
  );
}


__agency_exec_check_disable__
template<class Allocator, class ArrayView,
         __AGENCY_REQUIRES(
           !is_execution_policy<ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           bulk_destroy_detail::has_bulk_destroy_member<Allocator, ArrayView>::value
         )>
__AGENCY_ANNOTATION
void bulk_destroy(Allocator& alloc, ArrayView array)
{
  // call the allocator's member function
  alloc.bulk_destroy(array);
}


__agency_exec_check_disable__
template<class Allocator, class ArrayView,
         __AGENCY_REQUIRES(
           !is_execution_policy<ArrayView>::value
         ),
         __AGENCY_REQUIRES(
           !bulk_destroy_detail::has_bulk_destroy_member<Allocator, ArrayView>::value
         )>
__AGENCY_ANNOTATION
void bulk_destroy(Allocator& alloc, ArrayView array)
{
  // call destroy() in a loop
  for(size_t i = 0; i < array.size(); ++i)
  {
    agency::detail::allocator_traits<Allocator>::destroy(alloc, &array.begin()[i]);
  }
}


} // end detail
} // end agency

