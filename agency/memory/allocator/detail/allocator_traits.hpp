#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <memory>

namespace agency
{
namespace detail
{


template<class Alloc>
struct allocator_traits : std::allocator_traits<Alloc>
{
  using pointer = typename std::allocator_traits<Alloc>::pointer;
  using size_type = typename std::allocator_traits<Alloc>::size_type;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  static pointer allocate(Alloc& a, size_type n)
  {
    return a.allocate(n);
  }

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  static void deallocate(Alloc& a, pointer p, size_type n)
  {
    a.deallocate(p, n);
  }

  template<class T, class... Args>
  __AGENCY_ANNOTATION
  static void construct(Alloc& alloc, T* p, Args&&... args);

  template<class T>
  __AGENCY_ANNOTATION
  static void destroy(Alloc& a, T* p);

  __AGENCY_ANNOTATION
  static size_type max_size(const Alloc& a);
}; // end allocator_traits


} // end detail
} // end agency

#include <agency/memory/allocator/detail/allocator_traits/construct.hpp>
#include <agency/memory/allocator/detail/allocator_traits/destroy.hpp>
#include <agency/memory/allocator/detail/allocator_traits/max_size.hpp>

