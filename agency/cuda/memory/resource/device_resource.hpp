#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/device.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>


namespace agency
{
namespace cuda
{


class device_resource
{
  public:
    inline explicit device_resource(const device_id& d)
      : device_(d)
    {}

    inline device_resource()
      : device_resource(device_id(0))
    {}

    device_resource(const device_resource&) = default;

    inline void* allocate(size_t num_bytes)
    {
      // switch to our device
      scoped_device set_current_device(device());

      void* result = nullptr;
  
      cudaError_t error = cudaMalloc(&result, num_bytes);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "device_resource::allocate(): cudaMalloc");
      }
  
      return result;
    }

    inline void deallocate(void* ptr, size_t)
    {
      // switch to our device
      scoped_device set_current_device(device());

      cudaError_t error = cudaFree(ptr);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "device_resource::deallocate(): cudaFree");
      }
    }

    inline const device_id& device() const
    {
      return device_;
    }

    inline bool is_equal(const device_resource& other) const
    {
      return device() == other.device();
    }

  private:
    device_id device_;
};


inline bool operator==(const device_resource& a, const device_resource& b)
{
  return a.is_equal(b);
}

inline bool operator!=(const device_resource& a, const device_resource& b)
{
  return !(a == b);
}


} // end cuda
} // end agency

