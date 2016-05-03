#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/device.hpp>
#include <memory>


namespace agency
{
namespace cuda
{
namespace detail
{


class stream
{
  public:
    // creates a new stream associated with the current device
    inline __host__ __device__
    stream()
      : device_(current_device())
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaStreamCreate(&s_), "cudaStreamCreate in cuda::detail::stream ctor");
#else
      detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    // creates a new stream associated with the given device
    inline __host__ __device__
    stream(device_id g)
      : device_(g)
    {
      auto old_device = current_device();

#if __cuda_lib_has_cudart
#  ifdef __CUDA_ARCH__
      assert(g == old_device);
#  else
      detail::throw_on_error(cudaSetDevice(device().native_handle()), "cudaSetDevice in cuda::detail::stream ctor");
#  endif

      detail::throw_on_error(cudaStreamCreate(&s_), "cudaStreamCreate in cuda::detail::stream ctor");
#  ifdef __CUDA_ARCH__
      detail::throw_on_error(cudaSetDevice(old_device.native_handle()), "cudaSetDevice in cuda::detail::stream ctor");
#  endif
#else
      detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    inline __host__ __device__
    stream(stream&& other)
      : device_(other.device()), s_{}
    {
      s_ = other.s_;
      other.s_ = 0;
    }

    inline __host__ __device__
    ~stream()
    {
#if __cuda_lib_has_cudart
      if(s_ != 0)
      {
        // avoid propagating an exception but report the error if one exists
        detail::print_error_message_if(cudaStreamDestroy(s_), "cudaStreamDestroy in cuda::detail::stream dtor");
      }
#else
      detail::terminate_with_message("cuda::detail::stream dtor requires CUDART");
#endif
    }

    inline __host__ __device__
    device_id device() const
    {
      return device_;
    }

    inline __host__ __device__
    cudaStream_t native_handle() const
    {
      return s_;
    }

    inline __host__ __device__
    void swap(stream& other)
    {
      device_id tmp1 = device_;
      device_ = other.device_;
      other.device_ = tmp1;

      cudaStream_t tmp2 = s_;
      s_ = other.s_;
      other.s_ = tmp2;
    }

  private:
    static void callback(cudaStream_t, cudaError_t, void* user_data)
    {
      // XXX should maybe look at the CUDA error
      
      // convert user_data into a pointer to std::function and immediately put it inside unique_ptr
      std::unique_ptr<std::function<void()>> f_ptr(reinterpret_cast<std::function<void()>*>(user_data));

      // call f
      (*f_ptr)();
    }

  public:
    template<class Function>
    void add_callback(Function f)
    {
      // make a copy of f and put it inside a std::unique_ptr to std::function
      std::unique_ptr<std::function<void()>> ptr_to_fun(new std::function<void()>(f));

      // release the unique_ptr's pointer into cudaStreamAddCallback()
      detail::throw_on_error(cudaStreamAddCallback(native_handle(), callback, ptr_to_fun.release(), 0), "cudaStreamAddCallback in cuda::detail::stream::add_callback()");
    }

  private:
    device_id device_;
    cudaStream_t s_;
};


} // end detail
} // end cuda
} // end agency

