#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <vector>

namespace agency
{
namespace cuda
{


class device_id;


namespace detail
{


__host__ __device__
void set_current_device(const device_id& d);


__host__ __device__
device_id current_device();


// the CUDA Runtime's current device becomes the given device
// for as long as this object is in scope
class scoped_current_device
{
  public:
    scoped_current_device(const device_id& new_device);
    ~scoped_current_device();

  private:
    int old_device;
};


} // end detail


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

    void wait() const
    {
      detail::scoped_current_device temporary_device(*this);

#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaDeviceSynchronize(), "cuda::device_id::wait(): cudaDeviceSynchronize()");
#endif
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
void set_current_device(const device_id& d)
{
#if __cuda_lib_has_cudart
  throw_on_error(cudaSetDevice(d.native_handle()), "cuda::detail::set_current_device(): cudaSetDevice()");
#endif
}


__host__ __device__
device_id current_device()
{
  int result = -1;

#if __cuda_lib_has_cudart
  throw_on_error(cudaGetDevice(&result), "cuda::detail::current_device(): cudaGetDevice()");
#endif

  return device_id(result);
}
    

scoped_current_device::scoped_current_device(const device_id& new_device)
  : old_device(detail::current_device().native_handle())
{
  detail::set_current_device(new_device);
}


scoped_current_device::~scoped_current_device()
{
  detail::set_current_device(device_id(old_device));
}


std::vector<device_id> all_devices()
{
  std::vector<device_id> result;

#if __cuda_lib_has_cudart
  int device_count = 0;
  throw_on_error(cudaGetDeviceCount(&device_count), "cuda::detail::all_devices(): cudaGetDeviceCount()");
  result.reserve(static_cast<size_t>(device_count));

  for(int i = 0; i < device_count; ++i)
  {
    result.push_back(device_id(i));
  }
#endif

  return std::move(result);
}

template<class Container>
void wait(const Container& devices)
{
  for(auto& d : devices)
  {
    d.wait();
  }
}


__host__ __device__
size_t number_of_multiprocessors(const device_id& d)
{
#if __cuda_lib_has_cudart
  int attr = 0;
  throw_on_error(cudaDeviceGetAttribute(&attr, cudaDevAttrMultiProcessorCount, d.native_handle()), "cuda::detail::number_of_multiprocessors(): cudaDeviceGetAttribute()");
  return static_cast<size_t>(attr);
#else
  throw_on_error(cudaErrorNotSupported, "cuda::detail::number_of_multiprocessors(): cudaDeviceGetAttribute() requires CUDART");
  return 0;
#endif
}


__host__ __device__
size_t maximum_grid_size_x(const device_id& d)
{
#if __cuda_lib_has_cudart
  int attr = 0;
  throw_on_error(cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimX, d.native_handle()), "cuda::detail::maximum_grid_size(): cudaDeviceGetAttribute()");
  return static_cast<size_t>(attr);
#else
  throw_on_error(cudaErrorNotSupported, "cuda::detail::maximum_grid_size(): cudaDeviceGetAttribute() requires CUDART");
  return 0;
#endif
}


} // end detail
} // end cuda
} // end agency

