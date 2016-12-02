#pragma once

#include <agency/detail/config.hpp>
#include <stdlib.h>
#include <utility>

namespace agency
{
namespace detail
{


struct malloc_resource
{
  __AGENCY_ANNOTATION
  inline void* allocate(size_t num_bytes)
  {
    return malloc(num_bytes);
  }

  __AGENCY_ANNOTATION
  inline void deallocate(void* ptr, size_t)
  {
    free(ptr);
  }

  __AGENCY_ANNOTATION
  inline bool is_equal(const malloc_resource&) const
  {
    return true;
  }
};


__AGENCY_ANNOTATION
inline bool operator==(const malloc_resource& a, const malloc_resource& b)
{
  return a.is_equal(b);
}

__AGENCY_ANNOTATION
inline bool operator!=(const malloc_resource& a, const malloc_resource& b)
{
  return !(a == b);
}


} // end detail
} // end agency

