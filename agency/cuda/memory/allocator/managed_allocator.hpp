#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_adaptor.hpp>
#include <agency/cuda/memory/resource/managed_resource.hpp>
#include <agency/cuda/device.hpp>

namespace agency
{
namespace cuda
{


template<class T>
class managed_allocator : public agency::detail::allocator_adaptor<T,managed_resource>
{
  private:
    using super_t = agency::detail::allocator_adaptor<T,managed_resource>;

  public:
    explicit managed_allocator(const device_id& d)
      : super_t(managed_resource(d))
    {}

    managed_allocator()
      : managed_allocator(device_id(0))
    {}

    managed_allocator(const managed_allocator&) = default;

    template<class U>
    managed_allocator(const managed_allocator<U>& other)
      : managed_allocator(other.device())
    {}
};


} // end cuda
} // end agency

