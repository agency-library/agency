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


// XXX probably shouldn't inherit from thrust::cuda::pointer<T>
template<class T>
class unique_ptr : public thrust::cuda::pointer<T>
{
  private:
    using super_t = thrust::cuda::pointer<T>;

  public:
    using element_type = std::decay_t<T>;

    __host__ __device__
    explicit unique_ptr(thrust::cuda::pointer<T> ptr)
      : super_t(ptr)
    {}
  
    __host__ __device__
    unique_ptr(unique_ptr&& other)
      : super_t()
    {
      thrust::swap(static_cast<super_t&>(*this),
                   static_cast<super_t&>(other));
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
        thrust::cuda::free(*static_cast<super_t*>(this));
      }
#else
      __terminate_with_message("outer_shared_ptr dtor: thrust::cuda::free requires CUDART");
#endif
    }

    __host__ __device__
    unique_ptr& operator=(unique_ptr&& other)
    {
      thrust::swap(static_cast<super_t&>(*this),
                   static_cast<super_t&>(other));
      return *this;
    }

    __host__ __device__
    element_type* release()
    {
      element_type* result = super_t::get();
      *static_cast<super_t*>(this) = super_t{};
      return result;
    }
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

