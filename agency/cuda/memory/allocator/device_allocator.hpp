#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_adaptor.hpp>
#include <agency/cuda/memory/resource/device_resource.hpp>
#include <agency/cuda/device.hpp>

namespace agency
{
namespace cuda
{


template<class T>
class device_allocator : public agency::detail::allocator_adaptor<T,device_resource>
{
  private:
    using super_t = agency::detail::allocator_adaptor<T,device_resource>;

  public:
    explicit device_allocator(const device_id& device)
      : super_t(device_resource(device))
    {}

    device_allocator()
      : device_allocator(device_id(0))
    {}

    device_allocator(const device_allocator&) = default;

    template<class U>
    device_allocator(const device_allocator<U>& other)
      : device_allocator(other.device())
    {}
};


} // end cuda
} // end agency

