#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/memory/allocator_traits/check_for_member_functions.hpp>
#include <memory>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


__agency_exec_check_disable__
template<class Alloc, class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_destroy<Alloc,T>::value
>::type
  destroy(Alloc& a, T* p)
{
  a.destroy(p);
} // end destroy()


template<class Alloc, class T>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_destroy<Alloc,T*>::value
>::type
  destroy(Alloc& a, T* p)
{
  p->~T();
} // end destroy()


} // end allocator_traits_detail


template<class Alloc>
  template<class T>
__AGENCY_ANNOTATION
void allocator_traits<Alloc>
  ::destroy(Alloc& alloc, T* p)
{
  allocator_traits_detail::destroy(alloc, p);
} // end allocator_traits::destroy()


} // end detail
} // end agency

