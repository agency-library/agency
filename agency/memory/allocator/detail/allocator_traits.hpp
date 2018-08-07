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

  template<class Pointer, class... Args>
  __AGENCY_ANNOTATION
  static void construct(Alloc& alloc, Pointer p, Args&&... args);

  template<class Index, class Shape, class Pointer, class... Args>
  __AGENCY_ANNOTATION
  static void construct_array_element(Alloc& alloc, const Index& idx, const Shape& shape, Pointer p, Args&&... args);

  template<class Pointer>
  __AGENCY_ANNOTATION
  static void destroy(Alloc& a, Pointer p);

  template<class Index, class Shape, class Pointer>
  __AGENCY_ANNOTATION
  static void destroy_array_element(Alloc& a, const Index& idx, const Shape& shape, Pointer p);

  __AGENCY_ANNOTATION
  static size_type max_size(const Alloc& a);
}; // end allocator_traits


} // end detail
} // end agency

#include <agency/memory/allocator/detail/allocator_traits/construct.hpp>
#include <agency/memory/allocator/detail/allocator_traits/construct_array_element.hpp>
#include <agency/memory/allocator/detail/allocator_traits/destroy.hpp>
#include <agency/memory/allocator/detail/allocator_traits/destroy_array_element.hpp>
#include <agency/memory/allocator/detail/allocator_traits/max_size.hpp>

