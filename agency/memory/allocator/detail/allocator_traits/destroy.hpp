#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>
#include <agency/detail/requires.hpp>
#include <memory>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


__agency_exec_check_disable__
template<class Alloc, class Pointer,
         __AGENCY_REQUIRES(has_destroy<Alloc,Pointer>::value)
        >
__AGENCY_ANNOTATION
void destroy(Alloc& a, Pointer p)
{
  a.destroy(p);
} // end destroy()


__agency_exec_check_disable__
template<class Alloc, class Pointer,
         __AGENCY_REQUIRES(!has_destroy<Alloc,Pointer>::value)
        >
__AGENCY_ANNOTATION
void destroy(Alloc&, Pointer p)
{
  using element_type = typename std::pointer_traits<Pointer>::element_type;

  p->~element_type();
} // end destroy()


} // end allocator_traits_detail


template<class Alloc>
  template<class Pointer>
__AGENCY_ANNOTATION
void allocator_traits<Alloc>
  ::destroy(Alloc& alloc, Pointer p)
{
  allocator_traits_detail::destroy(alloc, p);
} // end allocator_traits::destroy()


} // end detail
} // end agency

