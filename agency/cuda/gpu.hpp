#pragma once

#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>

namespace agency
{
namespace cuda
{


class gpu_id
{
  public:
    typedef int native_handle_type;

    __host__ __device__
    gpu_id(native_handle_type handle)
      : handle_(handle)
    {}

    // default constructor creates a gpu_id which represents no gpu
    __host__ __device__
    gpu_id()
      : gpu_id(-1)
    {}

    // XXX std::this_thread::native_handle() is not const -- why?
    __host__ __device__
    native_handle_type native_handle() const
    {
      return handle_;
    }

    __host__ __device__
    friend inline bool operator==(gpu_id lhs, const gpu_id& rhs)
    {
      return lhs.handle_ == rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator!=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ != rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ < rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ <= rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ > rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>=(gpu_id lhs, gpu_id rhs)
    {
      return lhs.handle_ >= rhs.handle_;
    }

    friend std::ostream& operator<<(std::ostream &os, const gpu_id& id)
    {
      return os << id.native_handle();
    }

  private:
    native_handle_type handle_;
};


namespace detail
{


__host__ __device__
gpu_id current_gpu()
{
  int result = -1;

#if __cuda_lib_has_cudart
  throw_on_error(cudaGetDevice(&result), "cuda::detail::current_gpu(): cudaGetDevice()");
#endif

  return gpu_id(result);
}


} // end detail
} // end cuda
} // end agency

