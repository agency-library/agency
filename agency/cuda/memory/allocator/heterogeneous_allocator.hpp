#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/memory/resource/malloc_resource.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/memory/allocator/allocator_adaptor.hpp>
#include <agency/cuda/memory/resource/heterogeneous_resource.hpp>
#include <agency/cuda/memory/resource/managed_resource.hpp>
#include <memory>

namespace agency
{
namespace cuda
{


// heterogeneous_allocator uses a different primitive resource depending on whether
// its operations are called from __host__ or __device__ code.
template<class T, class HostResource = managed_resource, class DeviceResource = agency::detail::malloc_resource>
class heterogeneous_allocator : public agency::detail::allocator_adaptor<T,heterogeneous_resource<HostResource,DeviceResource>>
{
  private:
    using super_t = agency::detail::allocator_adaptor<T,heterogeneous_resource<HostResource,DeviceResource>>;

  public:
    __host__ __device__
    heterogeneous_allocator() = default;

    __host__ __device__
    heterogeneous_allocator(const heterogeneous_allocator&) = default;

    template<class U>
    __host__ __device__
    heterogeneous_allocator(const heterogeneous_allocator<U>& other)
      : super_t(other)
    {}
}; // end heterogeneous_allocator

template <typename T, typename U, typename HostResource, typename DeviceResource>
inline bool operator == (const heterogeneous_allocator<T,HostResource,DeviceResource>&, const heterogeneous_allocator<U,HostResource,DeviceResource>&)
{
      return true;
}

template <typename T, typename U, typename HostResource1, typename DeviceResource1, typename HostResource2, typename DeviceResource2>
inline bool operator == (const heterogeneous_allocator<T,HostResource1,DeviceResource1>&, const heterogeneous_allocator<U,HostResource2,DeviceResource2>&)
{
      return false;
}

template <typename T, typename U, typename HostResource1, typename DeviceResource1, typename HostResource2, typename DeviceResource2>
inline bool operator != (const heterogeneous_allocator<T,HostResource1,DeviceResource1>& a, const heterogeneous_allocator<U,HostResource2,DeviceResource2>& b)
{
      return !(a == b);
}

} // end cuda
} // end agency

