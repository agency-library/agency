#pragma once

#include <memory>
#include <type_traits>
#include <agency/detail/type_traits.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/cuda/detail/allocator.hpp>

// XXX should eliminate this dependency on Thrust
#include <thrust/detail/swap.h>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
class default_delete
{
  public:
    __host__ __device__
    default_delete(cudaStream_t s)
      : stream_(s)
    {}

    __host__ __device__
    default_delete() : default_delete(cudaStreamPerThread) {}

    __host__ __device__
    cudaStream_t stream() const
    {
      return stream_;
    }

    __host__ __device__
    void operator()(T* ptr) const
    {
      // destroy the object
      // XXX should use allocator_traits::destroy()
      ptr->~T();

      // deallocate
      allocator<T> alloc;
      alloc.deallocate(ptr, 1);
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
    unique_ptr(pointer ptr, const deleter_type& deleter = default_delete<T>())
      : ptr_(ptr),
        deleter_(deleter)
    {}

    __host__ __device__
    unique_ptr() : unique_ptr(nullptr) {}
  
    __host__ __device__
    unique_ptr(unique_ptr&& other)
      : ptr_(),
        deleter_(std::move(other.get_deleter()))
    {
      thrust::swap(ptr_, other.ptr_);
      thrust::swap(deleter_, other.deleter_);
    }
  
    __host__ __device__
    ~unique_ptr()
    {
      reset(nullptr);
    }

    __host__ __device__
    unique_ptr& operator=(unique_ptr&& other)
    {
      thrust::swap(ptr_,     other.ptr_);
      thrust::swap(deleter_, other.deleter_);
      return *this;
    }

    __host__ __device__
    pointer get() const
    {
      return ptr_;
    }

    __host__ __device__
    pointer release()
    {
      pointer result = nullptr;
      thrust::swap(ptr_, result);
      return result;
    }

    __host__ __device__
    void reset(pointer ptr = pointer())
    {
      pointer old_ptr = ptr_;
      ptr_ = ptr;

      if(old_ptr != nullptr)
      {
        get_deleter()(old_ptr); 
      }
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

    __host__ __device__
    const T& operator*() const
    {
      return *ptr_;
    }

    __host__ __device__
    T& operator*()
    {
      return *ptr_;
    }

    __host__ __device__
    operator bool () const
    {
      return get();
    }

    __host__ __device__
    void swap(unique_ptr& other)
    {
      thrust::swap(ptr_, other.ptr_);
      thrust::swap(deleter_, other.deleter_);
    }

  private:
    T* ptr_;
    default_delete<T> deleter_;
};


template<class T, class... Args>
__host__ __device__
unique_ptr<T> make_unique(cudaStream_t s, Args&&... args)
{
  allocator<T> alloc;

  unique_ptr<T> result(alloc.allocate(1), default_delete<T>(s));

  // XXX should use allocator_traits::construct()
  alloc.template construct<T>(result.get(), std::forward<Args>(args)...);

  return std::move(result);
}


} // end detail
} // end cuda
} // end agency

