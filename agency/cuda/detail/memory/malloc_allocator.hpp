#pragma once

#include <agency/detail/config.hpp>


namespace agency
{
namespace cuda
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

  template<class U, class... Args>
  __AGENCY_ANNOTATION
  void construct(U* ptr, Args&&... args)
  {
    ::new(ptr) T(std::forward<Args>(args)...);
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


template<>
struct malloc_allocator<void>
{
  using value_type = void;
};


} // end detail
} // end cuda
} // end agency

