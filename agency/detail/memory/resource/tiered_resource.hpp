#pragma once

#include <agency/detail/config.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{


template<class MemoryResource1, class MemoryResource2>
class tiered_resource : private MemoryResource1, private MemoryResource2
{
  public:
    using primary_resource_type = MemoryResource1;
    using fallback_resource_type = MemoryResource2;

    __AGENCY_ANNOTATION
    void* allocate(std::size_t n)
    {
      void* result = primary_resource_type::allocate(n);
      if(!result)
      {
        result = fallback_resource_type::allocate(n);
      }

      return result;
    }

    __AGENCY_ANNOTATION
    void deallocate(void* ptr, std::size_t n)
    {
      if(primary_resource_type::owns(ptr, n))
      {
        primary_resource_type::deallocate(ptr, n);
      }
      else
      {
        fallback_resource_type::deallocate(ptr, n);
      }
    }

    __AGENCY_ANNOTATION
    bool owns(void* ptr, std::size_t n) const
    {
      return primary_resource_type::owns(ptr, n) || fallback_resource_type::owns(ptr, n);
    }

    __AGENCY_ANNOTATION
    bool operator==(const tiered_resource& other) const
    {
      return static_cast<const primary_resource_type&>(*this) == static_cast<const primary_resource_type&>(other) &&
             static_cast<const fallback_resource_type&>(*this) == static_cast<const fallback_resource_type&>(other);
    }
};


} // end detail
} // end agency

