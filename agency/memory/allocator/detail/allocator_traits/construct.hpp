#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/memory/allocator/detail/allocator_traits/check_for_member_functions.hpp>
#include <agency/detail/iterator/forwarding_iterator.hpp>
#include <memory>

namespace agency
{
namespace detail
{
namespace allocator_traits_detail
{


__agency_exec_check_disable__
template<class Alloc, class Pointer, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_construct<Alloc,Pointer,Args...>::value
>::type
  construct(Alloc& a, Pointer p, Args&&... args)
{
  a.construct(p, std::forward<Args>(args)...);
} // end construct()


__agency_exec_check_disable__
template<class Alloc, class Pointer, class... Args>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_construct<Alloc,Pointer,Args...>::value
>::type
  construct(Alloc&, Pointer p, Args&&... args)
{
  using element_type = typename std::pointer_traits<Pointer>::element_type;

  ::new(p) element_type(std::forward<Args>(args)...);
} // end construct()


} // end allocator_traits_detail


template<class Alloc>
  template<class Pointer, class... Args>
__AGENCY_ANNOTATION
void allocator_traits<Alloc>
  ::construct(Alloc& alloc, Pointer p, Args&&... args)
{
  allocator_traits_detail::construct(alloc, p, std::forward<Args>(args)...);
} // end allocator_traits::construct()


} // end detail
} // end agency

