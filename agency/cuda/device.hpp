#pragma once

#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>

namespace agency
{
namespace cuda
{


class device_id
{
  public:
    typedef int native_handle_type;

    __host__ __device__
    device_id(native_handle_type handle)
      : handle_(handle)
    {}

    // default constructor creates a device_id which represents no device
    __host__ __device__
    device_id()
      : device_id(-1)
    {}

    // XXX std::this_thread::native_handle() is not const -- why?
    __host__ __device__
    native_handle_type native_handle() const
    {
      return handle_;
    }

    __host__ __device__
    friend inline bool operator==(device_id lhs, const device_id& rhs)
    {
      return lhs.handle_ == rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator!=(device_id lhs, device_id rhs)
    {
      return lhs.handle_ != rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<(device_id lhs, device_id rhs)
    {
      return lhs.handle_ < rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator<=(device_id lhs, device_id rhs)
    {
      return lhs.handle_ <= rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>(device_id lhs, device_id rhs)
    {
      return lhs.handle_ > rhs.handle_;
    }

    __host__ __device__
    friend inline bool operator>=(device_id lhs, device_id rhs)
    {
      return lhs.handle_ >= rhs.handle_;
    }

    friend std::ostream& operator<<(std::ostream &os, const device_id& id)
    {
      return os << id.native_handle();
    }

  private:
    native_handle_type handle_;
};


namespace detail
{


__host__ __device__
device_id current_device()
{
  int result = -1;

#if __cuda_lib_has_cudart
  throw_on_error(cudaGetDevice(&result), "cuda::detail::current_device(): cudaGetDevice()");
#endif

  return device_id(result);
}


} // end detail
} // end cuda
} // end agency

