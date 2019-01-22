#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/cuda/device.hpp>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <cuda_runtime.h>


namespace agency
{
namespace cuda
{


class managed_resource
{
  private:
    template<class... Args>
    static void swallow(Args&&...)
    {
    }

  public:
    inline explicit managed_resource(const device_id& d)
      : device_(d)
    {}

    inline managed_resource()
      : managed_resource(cuda::device(0))
    {}

    managed_resource(const managed_resource&) = default;

    inline void* allocate(size_t num_bytes)
    {
      // switch to our device
      scoped_device set_current_device(device());

      void* result = nullptr;
  
      cudaError_t error = cudaMallocManaged(&result, num_bytes, cudaMemAttachGlobal);
  
      if(error != cudaSuccess)
      {
        throw thrust::system_error(error, thrust::cuda_category(), "managed_resource::allocate(): cudaMallocManaged");
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
        throw thrust::system_error(error, thrust::cuda_category(), "managed_resource::deallocate(): cudaFree");
      }
    }

    inline const device_id& device() const
    {
      return device_;
    }

    inline bool is_equal(const managed_resource& other) const
    {
      return device() == other.device();
    }

  private:
    device_id device_;
};


inline bool operator==(const managed_resource& a, const managed_resource& b)
{
  return a.is_equal(b);
}

inline bool operator!=(const managed_resource& a, const managed_resource& b)
{
  return !(a == b);
}

inline bool operator<(const managed_resource& a, const managed_resource& b)
{
  return a.device() < b.device();
}


} // end cuda
} // end agency

