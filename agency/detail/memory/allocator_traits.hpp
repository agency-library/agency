#pragma once

#include <agency/detail/config.hpp>
#include <memory>

namespace agency
{
namespace detail
{


template<class Alloc>
struct allocator_traits : std::allocator_traits<Alloc>
{
  template<class T, class... Args>
  __AGENCY_ANNOTATION
  static void construct(Alloc& alloc, T* p, Args&&... args);

  template<class Iterator, class... Iterators>
  __AGENCY_ANNOTATION
  static Iterator construct_each(Alloc& alloc, Iterator first, Iterator last, Iterators... iters);
}; // end allocator_traits


} // end detail
} // end agency

#include <agency/detail/memory/allocator_traits/construct.hpp>
#include <agency/detail/memory/allocator_traits/construct_each.hpp>

