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
  Allocator alloc_;

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

    // make a copy of alloc because allocator_traits::construct_array_element() requires a mutable allocator
    Allocator mutable_alloc = alloc_;

    detail::allocator_traits<Allocator>::construct_array_element(mutable_alloc, idx, self.group_shape(), &array[idx], arrays[idx]...);
  }
};


template<class T, class Index>
struct is_indexable
{
  private:
    template<class U,
             class = decltype(std::declval<T>()[std::declval<Index>()])
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


} // end construct_array_detail


__agency_exec_check_disable__
template<class ExecutionPolicy, class Allocator, class ArrayView, class... ArrayViews,
         __AGENCY_REQUIRES(
           construct_array_detail::construct_array_requirements<decay_t<ExecutionPolicy>, ArrayView, ArrayViews...>::value
         )>
__AGENCY_ANNOTATION
void construct_array(ExecutionPolicy&& policy, Allocator& alloc, ArrayView array, ArrayViews... arrays)
{
  agency::bulk_invoke(
    policy(array.shape()),
    construct_array_detail::construct_array_functor<Allocator>{alloc},
    array,
    arrays...
  );
}


} // end detail
} // end agency

