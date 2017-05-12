#pragma once

#include <agency/detail/config.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{


// a null_resource is a memory resource that always fails to allocate
// unlike arena_resource, it does not throw an exception or otherwise report error upon failure
// besides returning nullptr from .allocate()
// null_resource is useful in contexts where no dynamic memory is not needed and error reporting
// is expensive
struct null_resource
{
  __AGENCY_ANNOTATION
  void* allocate(std::size_t) noexcept
  {
    return nullptr;
  }

  __AGENCY_ANNOTATION
  void deallocate(void*, std::size_t) noexcept
  {
  }
};


} // end detail
} // end agency

