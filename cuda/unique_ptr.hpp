#pragma once

#include <memory>
#include <thrust/system/cuda/memory.h>
#include <thrust/detail/swap.h>
#include <type_traits>
#include "terminate.hpp"


namespace cuda
{
namespace detail
{


struct destroy_functor
{
  template<class T>
  __host__ __device__
  void operator()(T& x)
  {
    x.~T();
  }
};


}


template<class T>
class unique_ptr
{
  public:
    using element_type = std::decay_t<T>;

    __host__ __device__
    explicit unique_ptr(thrust::cuda::pointer<T> ptr)
      : ptr_(ptr)
    {}
  
    __host__ __device__
    unique_ptr(unique_ptr&& other)
      : ptr_()
    {
      thrust::swap(ptr_, other.ptr_);
    }
  
    __host__ __device__
    ~unique_ptr()
    {
#if __cuda_lib_has_cudart
      if(this->get())
      {
        // call T's destructor
        // XXX we should do this with bulk_invoke
        thrust::for_each(thrust::cuda::par, this->get(), this->get() + 1, detail::destroy_functor());

        // deallocate
        thrust::cuda::free(ptr_);
      }
#else
      __terminate_with_message("outer_shared_ptr dtor: thrust::cuda::free requires CUDART");
#endif
    }

    __host__ __device__
    unique_ptr& operator=(unique_ptr&& other)
    {
      thrust::swap(ptr_, other.ptr_);
      return *this;
    }

    __host__ __device__
    element_type* get() const
    {
      return ptr_.get();
    }

    __host__ __device__
    element_type* release()
    {
      thrust::cuda::pointer<element_type> result;
      thrust::swap(ptr_, result);
      return result.get();
    }

  private:
    thrust::cuda::pointer<T> ptr_;
};


// XXX need to take parameters and call constructor
template<class T>
__host__ __device__
unique_ptr<T> make_unique()
{
  unique_ptr<T> result(thrust::cuda::malloc<T>(1));

  // XXX call constructor here

  return std::move(result);
}


} // end cuda

