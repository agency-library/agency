#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/allocator_traits.hpp>
#include <agency/tuple.hpp>


namespace agency
{
namespace cuda
{


// heterogeneous_resource uses a different primitive resource depending on whether
// its operations are called from __host__ or __device__ code.
template<class HostResource, class DeviceResource>
class heterogeneous_resource
{
  public:
    using host_resource = HostResource;
    using device_resource = DeviceResource;

    __agency_exec_check_disable__
    __host__ __device__
    heterogeneous_resource() :
#ifndef __CUDA_ARCH__
      host_resource_{}
#else
      device_resource_{}
#endif
    {}

    __host__ __device__
    void* allocate(size_t num_bytes)
    {
#ifndef __CUDA_ARCH__
      return host_resource_.allocate(num_bytes);
#else
      return device_resource_.allocate(num_bytes);
#endif
    }

    __host__ __device__
    void deallocate(void* ptr, size_t num_bytes)
    {
#ifndef __CUDA_ARCH__
      return host_resource_.deallocate(ptr, num_bytes);
#else
      return device_resource_.deallocate(ptr, num_bytes);
#endif
    }

    __host__ __device__
    bool operator==(const heterogeneous_resource& other) const
    {
#ifndef __CUDA_ARCH__
      return host_resource_ == other.host_resource_;
#else
      return device_resource_ == other.device_resource_;
#endif
    }

    __host__ __device__
    bool operator!=(const heterogeneous_resource& other) const
    {
#ifndef __CUDA_ARCH__
      return host_resource_ != other.host_resource_;
#else
      return device_resource_ != other.device_resource_;
#endif
    }

  private:
#ifndef __CUDA_ARCH__
    host_resource host_resource_;
#else
    device_resource device_resource_;
#endif
};


} // end cuda
} // end agency

