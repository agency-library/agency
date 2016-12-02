#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/memory/allocator/allocator_adaptor.hpp>
#include <agency/cuda/memory/resource/device_resource.hpp>
#include <agency/cuda/device.hpp>
#include <cuda_runtime.h>

namespace agency
{
namespace cuda
{


template<class T>
class device_allocator : public agency::detail::allocator_adaptor<T,device_resource>
{
  // XXX since construct_n() doesn't actually do anything, it might
  //     be a good idea to static_assert that T is trivially constructible

  private:
    using super_t = agency::detail::allocator_adaptor<T,device_resource>;

  public:
    explicit device_allocator(const device_id& device)
      : super_t(device_resource(device))
    {}

    device_allocator()
      : device_allocator(detail::all_devices()[0])
    {}

    device_allocator(const device_allocator&) = default;

    template<class U>
    device_allocator(const device_allocator<U>& other)
      : device_allocator(other.device())
    {}
  
    // XXX we should hoist this up into the memory resource 
    // XXX this should be implemented with a kernel launch or something
    template<class Iterator, class... Iterators>
    agency::detail::tuple<Iterator,Iterators...> construct_n(Iterator first, size_t n, Iterators... iters)
    {
      //new(ptr) U(*iters...);
      return agency::detail::make_tuple(first,iters...);
    }
};


} // end cuda
} // end agency

