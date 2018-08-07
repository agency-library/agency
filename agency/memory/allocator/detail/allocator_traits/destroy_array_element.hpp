#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>
#include <memory>
#include <utility>


namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


__agency_exec_check_disable__
template<class Alloc, class Index, class Shape, class Pointer,
         __AGENCY_REQUIRES(
           has_destroy_array_element<Alloc,Index,Shape,Pointer>::value
         )>
__AGENCY_ANNOTATION
void destroy_array_element(Alloc& a, const Index& idx, const Shape& shape, Pointer p)
{
  a.destroy(idx, shape, p);
}

__agency_exec_check_disable__
template<class Alloc, class Index, class Shape, class Pointer,
         __AGENCY_REQUIRES(
           !has_destroy_array_element<Alloc,Index,Shape,Pointer>::value
         )>
__AGENCY_ANNOTATION
void destroy_array_element(Alloc& a, const Index&, const Shape&, Pointer p)
{
  // ignore the index and shape and use normal destroy()
  allocator_traits<Alloc>::destroy(a, p);
}


} // end allocator_traits_detail


template<class Alloc>
  template<class Index, class Shape, class Pointer>
__AGENCY_ANNOTATION
void allocator_traits<Alloc>
  ::destroy_array_element(Alloc& alloc, const Index& idx, const Shape& shape, Pointer p)
{
  allocator_traits_detail::destroy_array_element(alloc, idx, shape, p);
}


} // end detail
} // end agency

