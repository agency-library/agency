#pragma once

#include <memory>
#include <agency/detail/type_traits.hpp>
#include "launch_kernel.hpp"
#include "workaround_unused_variable_warning.hpp"

// XXX should eliminate this dependency on Thrust
#include <thrust/system/cuda/memory.h>
#include <thrust/detail/swap.h>


namespace cuda
{
namespace detail
{


template<class T>
__global__ void destroy_kernel(T* ptr)
{
  ptr->~T();
}

template<class T, class... Args>
__global__ void construct_kernel(T* ptr, Args... args)
{
  ::new(ptr) T(args...);
}


template<class T>
class default_delete
{
  public:
    __host__ __device__
    default_delete(cudaStream_t s)
      : stream_(s)
    {}

    __host__ __device__
    cudaStream_t stream() const
    {
      return stream_;
    }

    __host__ __device__
    void operator()(T* ptr) const
    {
      auto kernel = detail::destroy_kernel<T>;
      detail::workaround_unused_variable_warning(kernel);

#ifndef __CUDA_ARCH__
      // we're executing on the host; launch a kernel to call the destructor
      detail::launch_kernel(reinterpret_cast<void*>(kernel), uint2{1,1}, 0, stream(), ptr);
#else
      // we're executing on the device; just call the destructor directly
      ptr->~T();
#endif

      // deallocate
      thrust::cuda::free(thrust::cuda::pointer<T>(ptr));
    }

  private:
    cudaStream_t stream_;
};


template<class T>
class unique_ptr
{
  public:
    using element_type = agency::detail::decay_t<T>;
    using pointer      = element_type*;
    using deleter_type = default_delete<element_type>;

    __host__ __device__
    unique_ptr(thrust::cuda::pointer<T> ptr, const deleter_type& deleter = default_delete<T>())
      : ptr_(ptr),
        deleter_(deleter)
    {}
  
    __host__ __device__
    unique_ptr(unique_ptr&& other)
      : ptr_(),
        deleter_(std::move(other.get_deleter()))
    {
      thrust::swap(ptr_, other.ptr_);
    }
  
    __host__ __device__
    ~unique_ptr()
    {
      if(this->get())
      {
        get_deleter()(this->get());
      }
    }

    __host__ __device__
    unique_ptr& operator=(unique_ptr&& other)
    {
      thrust::swap(ptr_, other.ptr_);
      return *this;
    }

    __host__ __device__
    pointer get() const
    {
      return ptr_.get();
    }

    __host__ __device__
    pointer release()
    {
      thrust::cuda::pointer<element_type> result;
      thrust::swap(ptr_, result);
      return result.get();
    }

    __host__ __device__
    deleter_type& get_deleter()
    {
      return deleter_;
    }

    __host__ __device__
    const deleter_type& get_deleter() const
    {
      return deleter_;
    }

  private:
    thrust::cuda::pointer<T> ptr_;
    default_delete<T> deleter_;
};


template<class T, class... Args>
__host__ __device__
unique_ptr<T> make_unique(cudaStream_t s, Args&&... args)
{
  auto deleter = default_delete<T>(s);
  unique_ptr<T> result(thrust::cuda::malloc<T>(1), deleter);

  auto kernel = detail::construct_kernel<T,agency::detail::decay_t<Args>...>;
  detail::workaround_unused_variable_warning(kernel);

#ifndef __CUDA_ARCH__
  // we're executing on the host; launch a kernel to call the destructor
  detail::checked_launch_kernel(reinterpret_cast<void*>(kernel), uint2{1,1}, 0, s, result.get(), std::forward<Args>(args)...);
#else
  // we're executing on the device; just placement new directly
  ::new(result.get()) T(std::forward<Args>(args)...);
#endif

  return std::move(result);
}


} // end detail
} // end cuda

