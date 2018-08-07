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
template<class Alloc, class Index, class Shape, class Pointer, class... Args,
         __AGENCY_REQUIRES(
           has_construct_array_element<Alloc,Index,Shape,Pointer,Args...>::value
         )>
__AGENCY_ANNOTATION
void construct_array_element(Alloc& a, const Index& idx, const Shape& shape, Pointer p, Args&&... args)
{
  a.construct_array_element(idx, shape, p, std::forward<Args>(args)...);
}

__agency_exec_check_disable__
template<class Alloc, class Index, class Shape, class Pointer, class... Args,
         __AGENCY_REQUIRES(
           !has_construct_array_element<Alloc,Index,Shape,Pointer,Args...>::value
         )>
__AGENCY_ANNOTATION
void construct_array_element(Alloc& a, const Index&, const Shape&, Pointer p, Args&&... args)
{
  // ignore the index and shape and use normal construct()
  allocator_traits<Alloc>::construct(a, p, std::forward<Args>(args)...);
}


} // end allocator_traits_detail


template<class Alloc>
  template<class Index, class Shape, class Pointer, class... Args>
__AGENCY_ANNOTATION
void allocator_traits<Alloc>
  ::construct_array_element(Alloc& alloc, const Index& idx, const Shape& shape, Pointer p, Args&&... args)
{
  allocator_traits_detail::construct_array_element(alloc, idx, shape, p, std::forward<Args>(args)...);
}


} // end detail
} // end agency

