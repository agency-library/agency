#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
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
  static detail::tuple<Iterator,Iterators...> construct_n(Alloc& alloc, Iterator first, size_t n, Iterators... iters);
}; // end allocator_traits


} // end detail
} // end agency

#include <agency/detail/memory/allocator_traits/construct.hpp>
#include <agency/detail/memory/allocator_traits/construct_n.hpp>

