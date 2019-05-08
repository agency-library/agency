#pragma once

#include <agency/detail/config.hpp>
#include <agency/container/array.hpp>
#include <agency/container/vector.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <vector>
#include <algorithm>


namespace agency
{
namespace cuda
{


class device_id;


// the CUDA Runtime's current device becomes the given device
// for as long as this object is in scope
class scoped_device
{
  public:
    __host__ __device__
    inline scoped_device(const device_id& new_device);

    __host__ __device__
    inline ~scoped_device();

  private:
    int old_device;
};


namespace detail
{


__host__ __device__
inline void set_current_device(const device_id& d);


__host__ __device__
inline device_id current_device();


template<class T>
struct is_range_of_device_id_impl
{
  // this is hacky but sufficient for our purposes
  template<class U,

           // get U's iterator type
           class Iterator = decltype(std::declval<U>().begin()),

           // get U's sentinel type
           class Sentinel = decltype(std::declval<U>().end()),

           // get iterator's value_type
           class ValueType = typename std::iterator_traits<Iterator>::value_type,

           // ValueType should be device_id
           class Result = std::is_same<ValueType, device_id>
          >
  static Result test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<T>(0));
};


template<class T>
using is_range_of_device_id = typename is_range_of_device_id_impl<T>::type;


} // end detail


class device_id
{
  public:
    typedef int native_handle_type;

    __host__ __device__
    constexpr device_id(native_handle_type handle)
      : handle_(handle)
    {}

    // default constructor creates a device_id which represents no device
    __host__ __device__
    constexpr device_id()
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
      scoped_device temporary_device(*this);

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



// device() is included for symmetry with devices()
// note that when an integer is passed as a parameter to device(d),
// it will be automatically converted into a device_id
__AGENCY_ANNOTATION
inline device_id device(device_id d)
{
  return d;
}


template<class... IntegersOrDeviceIds>
__AGENCY_ANNOTATION
array<device_id, 1 + sizeof...(IntegersOrDeviceIds)> devices(device_id id0, IntegersOrDeviceIds... ids)
{
  return {{id0, device_id(ids)...}};
}


template<class Range,
         __AGENCY_REQUIRES(
           !std::is_convertible<const Range&, device_id>::value
         )>
__AGENCY_ANNOTATION
vector<device_id> devices(const Range& integers_or_device_ids)
{
  vector<device_id> result(integers_or_device_ids.size());

  for(size_t i = 0; i < integers_or_device_ids.size(); ++i)
  {
    result[i] = device_id(integers_or_device_ids[i]);
  }

  return result;
}


__AGENCY_ANNOTATION
inline vector<device_id> all_devices()
{
  vector<device_id> result;

#if __cuda_lib_has_cudart
  int device_count = 0;
  detail::throw_on_error(cudaGetDeviceCount(&device_count), "cuda::all_devices(): cudaGetDeviceCount()");
  result.reserve(static_cast<size_t>(device_count));

  for(int i = 0; i < device_count; ++i)
  {
    result.push_back(device_id(i));
  }
#endif

  return result;
}


namespace detail
{


__host__ __device__
inline void set_current_device(const device_id& d)
{
#if __cuda_lib_has_cudart
#ifndef __CUDA_ARCH__
  throw_on_error(cudaSetDevice(d.native_handle()), "cuda::detail::set_current_device(): cudaSetDevice()");
#else
  if(d != current_device())
  {
    detail::throw_on_error(cudaErrorNotSupported, "cuda::detail::set_current_device(): Unable to set a different device in __device__ code.");
  }
#endif // __CUDA_ARCH__
#endif // __cuda_lib_has_cudart
}


__host__ __device__
inline device_id current_device()
{
  int result = -1;

#if __cuda_lib_has_cudart
  throw_on_error(cudaGetDevice(&result), "cuda::detail::current_device(): cudaGetDevice()");
#endif

  return device_id(result);
}


} // end detail
    

__host__ __device__
scoped_device::scoped_device(const device_id& new_device)
  : old_device(detail::current_device().native_handle())
{
  detail::set_current_device(new_device);
}


__host__ __device__
scoped_device::~scoped_device()
{
  detail::set_current_device(device_id(old_device));
}


namespace detail
{


template<class Container>
void wait(const Container& devices)
{
  for(auto& d : devices)
  {
    d.wait();
  }
}


inline bool has_concurrent_managed_access(const device_id& device)
{
  int result = 0;

#if __cuda_lib_has_cudart
  throw_on_error(cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, device.native_handle()), "cuda::detail::has_concurrent_managed_access(): cudaDeviceGetAttribute()");
#endif

  return result;
}


template<class Container>
void wait_if_any_lack_concurrent_managed_access(const Container& devices)
{
  // if not all of the devices have concurrent managed access...
  if(!std::all_of(devices.begin(), devices.end(), has_concurrent_managed_access))
  {
    // then wait for all of the devices to become idle
    wait(devices);
  }
}


__host__ __device__
inline size_t number_of_multiprocessors(const device_id& d)
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
inline size_t maximum_grid_size_x(const device_id& d)
{
#if __cuda_lib_has_cudart
  int attr = 0;
  throw_on_error(cudaDeviceGetAttribute(&attr, cudaDevAttrMaxGridDimX, d.native_handle()), "cuda::detail::maximum_grid_size_x(): cudaDeviceGetAttribute()");
  return static_cast<size_t>(attr);
#else
  throw_on_error(cudaErrorNotSupported, "cuda::detail::maximum_grid_size_x(): cudaDeviceGetAttribute() requires CUDART");
  return 0;
#endif
}


__host__ __device__
inline size_t maximum_block_size_x(const device_id& d)
{
#if __cuda_lib_has_cudart
  int attr = 0;
  throw_on_error(cudaDeviceGetAttribute(&attr, cudaDevAttrMaxBlockDimX, d.native_handle()), "cuda::detail::maximum_block_size_x(): cudaDeviceGetAttribute()");
  return static_cast<size_t>(attr);
#else
  throw_on_error(cudaErrorNotSupported, "cuda::detail::maximum_block_size_x(): cudaDeviceGetAttribute() requires CUDART");
  return 0;
#endif
}


} // end detail
} // end cuda
} // end agency

