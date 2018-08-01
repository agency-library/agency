#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <utility>
#include <memory>

namespace agency
{
namespace detail
{


// allocator_adaptor adapts a memory resource that allocates bytes into an allocator that
// allocates objects
template<class T, class MemoryResource>
class allocator_adaptor : private MemoryResource
{
  private:
    using super_t = MemoryResource;

  public:
    using value_type = T;

    __agency_exec_check_disable__
    allocator_adaptor() = default;

    __agency_exec_check_disable__
    allocator_adaptor(const allocator_adaptor&) = default;

    __agency_exec_check_disable__
    template<class U>
    __AGENCY_ANNOTATION
    allocator_adaptor(const allocator_adaptor<U,MemoryResource>& other)
      : super_t(other.resource())
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    allocator_adaptor(const MemoryResource& resource)
      : super_t(resource)
    {}

    __agency_exec_check_disable__
    template<class U = T, __AGENCY_REQUIRES(!std::is_void<U>::value)>
    __AGENCY_ANNOTATION
    value_type *allocate(size_t n)
    {
      return reinterpret_cast<value_type*>(super_t::allocate(n * sizeof(T)));
    }

    __agency_exec_check_disable__
    template<class U = T, __AGENCY_REQUIRES(!std::is_void<U>::value)>
    __AGENCY_ANNOTATION
    void deallocate(value_type* ptr, size_t n)
    {
      super_t::deallocate(ptr, n * sizeof(T));
    }

    __AGENCY_ANNOTATION
    const MemoryResource& resource() const
    {
      return *this;
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bool operator==(const allocator_adaptor& other) const
    {
      return static_cast<const MemoryResource&>(*this) == static_cast<const MemoryResource&>(other);
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    bool operator!=(const allocator_adaptor& other) const
    {
      return static_cast<const MemoryResource&>(*this) != static_cast<const MemoryResource&>(other);
    }
};


} // end detail
} // end agency

