#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


class event
{
  public:
    struct construct_ready_t {};
    struct construct_not_ready_t {};

    static constexpr construct_ready_t construct_ready{};
    static constexpr construct_not_ready_t construct_not_ready{};

    static constexpr int event_create_flags = cudaEventDisableTiming;

    __host__ __device__
    event(cudaEvent_t e) : e_(e) {}

    __host__ __device__
    event() : event(0) {}

    __host__ __device__
    event(construct_not_ready_t) : event() {}

    __host__ __device__
    event(construct_ready_t)
      : event(construct_not_ready)
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&e_, event_create_flags), "cudaEventCreateWithFlags in cuda::detail::event ctor");
#else
      detail::terminate_with_message("cuda::detail::event ctor requires CUDART");
#endif
    }

    __host__ __device__
    event(const event&) = delete;

    __host__ __device__
    event(event&& other)
      : e_(other.e_)
    {
      other.e_ = 0;
    }

    __host__ __device__
    ~event()
    {
      if(valid())
      {
        destroy_event();
      }
    }

    __host__ __device__
    cudaEvent_t release()
    {
      cudaEvent_t result = e_;
      e_ = 0;

      return result;
    }

    // XXX eliminate this!
    __host__ __device__
    cudaEvent_t get() const
    {
      return e_;
    }

    // XXX eliminate this!
    __host__ __device__
    void destroy_event()
    {
#if __cuda_lib_has_cudart
      // since this will likely be called from destructors, swallow errors
      cudaError_t error = cudaEventDestroy(e_);
      e_ = 0;

#if __cuda_lib_has_printf
      if(error)
      {
        printf("CUDA error after cudaEventDestroy in cuda::detail::event::destroy_event: %s", cudaGetErrorString(error));
      } // end if
#endif // __cuda_lib_has_printf
#endif // __cuda_lib_has_cudart
    }

    __host__ __device__
    bool valid() const
    {
      return e_ != 0;
    }

    __host__ __device__
    void wait() const
    {
      // XXX should probably check for valid() here

#if __cuda_lib_has_cudart

#ifndef __CUDA_ARCH__
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::throw_on_error(cudaEventSynchronize(e_), "cudaEventSynchronize in cuda::detail::event::wait");
#else
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::throw_on_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize in cuda::detail::event::wait");
#endif // __CUDA_ARCH__

#else
      detail::terminate_with_message("cuda::detail::event::wait requires CUDART");
#endif // __cuda_lib_has_cudart
    }

    __host__ __device__
    void swap(event& other)
    {
      cudaEvent_t tmp = e_;
      e_ = other.e_;
      other.e_ = tmp;
    }

  private:
    cudaEvent_t e_;
};


} // end detail
} // end cuda
} // end agency

