#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/pointer_adaptor.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <cuda_runtime_api.h>
#include <stdexcept>


namespace agency
{
namespace cuda
{
namespace detail
{


class device_accessor
{
  private:
    template<class T>
    __AGENCY_ANNOTATION
    static T local_load(const T* ptr)
    {
      return *ptr;
    }

    template<class T>
    __AGENCY_ANNOTATION
    static int local_store(T* ptr, const T& value)
    {
      *ptr = value;
      return 0;
    }

  public:
    device_accessor() = default;
    device_accessor(const device_accessor&) = default;
    device_accessor& operator=(const device_accessor&) = default;

    // loads a value at a pointer in the device memory space
    template<class T>
    __AGENCY_ANNOTATION
    T load(const T* ptr) const
    {
#ifndef __CUDA_ARCH__
      T result{};
      agency::cuda::detail::throw_on_error(cudaMemcpy(&result, ptr, sizeof(T), cudaMemcpyDeviceToHost), "device_accessor::load");

      return result;
#else
      return device_accessor::local_load(ptr);
#endif
    }

    // immediate store to device memory from a value
    template<class T>
    __AGENCY_ANNOTATION
    void store(T* ptr, const T& value) const
    {
#ifndef __CUDA_ARCH__
      agency::cuda::detail::throw_on_error(cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice), "device_accessor::store");
#else
      device_accessor::local_store(ptr, value);
#endif
    }

    // indirect store to device memory from device memory
    template<class T>
    __AGENCY_ANNOTATION
    void store(T* dst, const T* src) const
    {
#ifndef __CUDA_ARCH__
      agency::cuda::detail::throw_on_error(cudaMemcpy(dst, src, sizeof(T), cudaMemcpyHostToDevice), "device_accessor::store");
#else
      device_accessor::local_store(dst, *src);
#endif
    }

    __AGENCY_ANNOTATION
    bool operator==(const device_accessor&) const noexcept
    {
      return true;
    }

    __AGENCY_ANNOTATION
    bool operator!=(const device_accessor&) const noexcept
    {
      return false;
    }
};


} // end detail


template<class T>
using device_ptr = pointer_adaptor<T, detail::device_accessor>;


template<class T>
using device_reference = pointer_adaptor_reference<T, detail::device_accessor>;


template<class T>
__AGENCY_ANNOTATION
T* to_address(const device_ptr<T>& ptr) noexcept
{
  return ptr.get();
}

template<class T>
__AGENCY_ANNOTATION
T* to_address(T* ptr) noexcept
{
  return ptr;
}


} // end cuda
} // end agency

