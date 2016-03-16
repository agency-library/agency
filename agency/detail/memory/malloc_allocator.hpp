#pragma once

#include <agency/detail/config.hpp>
#include <stdlib.h>
#include <utility>

namespace agency
{
namespace detail
{


template<class T>
struct malloc_allocator
{
  using value_type = T;

  __AGENCY_ANNOTATION
  malloc_allocator() = default;

  __AGENCY_ANNOTATION
  malloc_allocator(const malloc_allocator&) = default;

  template<class U>
  __AGENCY_ANNOTATION
  malloc_allocator(const malloc_allocator<U>&) {}

  __agency_hd_warning_disable__
  template<class U, class... Args>
  __AGENCY_ANNOTATION
  void construct(U* ptr, Args&&... args)
  {
    ::new(ptr) U(std::forward<Args>(args)...);
  }

  __AGENCY_ANNOTATION
  value_type* allocate(size_t n)
  {
    return reinterpret_cast<value_type*>(malloc(n * sizeof(T)));
  }

  __AGENCY_ANNOTATION
  void deallocate(value_type* ptr, size_t)
  {
    free(ptr);
  }
};


} // end detail
} // end agency

