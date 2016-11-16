#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/cuda/device.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace agency
{
namespace cuda
{


template<class T>
class managed_allocator
{
  private:
    template<class> friend class managed_allocator;

    device_id device_;

    const device_id& device() const
    {
      return device_;
    }

    template<class... Args>
    static void swallow(Args&&...)
    {
    }

  public:
    using value_type = T;

    explicit managed_allocator(const device_id& d)
      : device_(d)
    {}

    managed_allocator()
      : managed_allocator(detail::all_devices()[0])
    {}

    managed_allocator(const managed_allocator&) = default;

    template<class U>
    managed_allocator(const managed_allocator<U>& other)
      : device_(other.device_)
    {}
  
    value_type* allocate(size_t n)
    {
      // switch to our device
      detail::scoped_current_device set_current_device(device());

      value_type* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, n * sizeof(T), cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::allocate(): cudaMallocManaged");
      }
  
      return result;
    }
  
    void deallocate(value_type* ptr, size_t)
    {
      // switch to our device
      detail::scoped_current_device set_current_device(device());

      cudaError_t error = cudaFree(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_allocator::deallocate(): cudaFree");
      }
    }

    template<class Iterator, class... Iterators,
             __AGENCY_REQUIRES(
               std::is_trivially_constructible<
                 T,
                 typename std::iterator_traits<Iterator>::reference,
                 typename std::iterator_traits<Iterators>::reference...
               >::value
              )
            >
    agency::detail::tuple<Iterator,Iterators...> construct_n(Iterator, size_t, Iterators...)
    {
      // T's constructor has no effects, so this is a no-op
    }

    template<class Iterator, class... Iterators,
             __AGENCY_REQUIRES(
               !std::is_trivially_constructible<
                 T,
                 typename std::iterator_traits<Iterator>::reference,
                 typename std::iterator_traits<Iterators>::reference...
               >::value
              )
            >
    agency::detail::tuple<Iterator,Iterators...> construct_n(Iterator first, size_t n, Iterators... iters)
    {
      // we need to globally synchronize if the system contains any device that lack
      // concurrent managed access before the host can access managed memory
      detail::wait_if_any_lack_concurrent_managed_access(detail::all_devices());

      using value_type = typename std::iterator_traits<Iterator>::value_type;

      for(size_t i = 0; i < n; ++i, ++first, swallow(++iters...))
      {
        new(&*first) value_type(*iters...);
      }

      return agency::detail::make_tuple(first, iters...);
    }
};


} // end cuda
} // end agency

