#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/memory/allocator_traits.hpp>
#include <agency/detail/tuple.hpp>


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

    template<class Iterator, class... Iterators>
    __host__ __device__
    agency::detail::tuple<Iterator,Iterators...> construct_n(Iterator first, size_t n, Iterators... iters)
    {
#ifndef __CUDA_ARCH__
      return agency::detail::allocator_traits_detail::construct_n_impl1(host_resource_, first, n, iters...);
#else
      return agency::detail::allocator_traits_detail::construct_n_impl1(device_resource_, first, n, iters...);
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

