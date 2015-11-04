#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/kernel.hpp>
#include <agency/cuda/gpu.hpp>

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

    // constructs a new event recorded on the given stream
    __host__ __device__
    event(cudaStream_t s) : event(construct_ready)
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventRecord(e_, s), "cudaEventRecord in cuda::detail::event ctor");
#else
      detail::terminate_with_message("cuda::detail::event ctor requires CUDART");
#endif
    }

    __host__ __device__
    event() : event(cudaEvent_t{0}) {}

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
      detail::throw_on_error(cudaEventSynchronize(e_), "cudaEventSynchronize in cuda::detail::event::wait");
#else
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

    template<class... Args>
    __host__ __device__
    event then(void* kernel, dim3 grid_dim, dim3 block_dim, int shared_memory_size, cudaStream_t stream, const Args&... args)
    {
      // make the stream wait before launching
      stream_wait(stream);

      detail::checked_launch_kernel(kernel, grid_dim, block_dim, shared_memory_size, stream, args...);

      // invalidate ourself
      destroy_event();

      return event(stream);
    }

    template<class... Args>
    __host__ __device__
    event then_on(void* kernel, dim3 grid_dim, dim3 block_dim, int shared_memory_size, cudaStream_t stream, const gpu_id& gpu, const Args&... args)
    {
      // make the stream wait before launching
      stream_wait(stream);

      detail::checked_launch_kernel_on_device(kernel, grid_dim, block_dim, shared_memory_size, stream, gpu.native_handle(), args...);

      // invalidate ourself
      destroy_event();

      return event(stream);
    }

  private:
    cudaEvent_t e_;

    __host__ __device__
    event(cudaEvent_t e) : e_(e) {}

    // this function returns 0 so that we can pass it as an argument to swallow(...)
    __host__ __device__
    int destroy_event()
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

      return 0;
    }

    // this function returns 0 so that we can pass it as an argument to swallow(...)
    // XXX this function should really be a member of a stream class
    __host__ __device__
    int stream_wait(cudaStream_t s) const
    {
#if __cuda_lib_has_cudart
      // make the next launch wait on the event
      throw_on_error(cudaStreamWaitEvent(s, e_, 0), "cuda::detail::event::stream_wait(): cudaStreamWaitEvent()");
#else
      throw_on_error(cudaErrorNotSupported, "cuda::detail::event::stream_wait(): cudaStreamWaitEvent() requires CUDART");
#endif // __cuda_lib_has_cudart

      return 0;
    }

    template<class... Args>
    inline __host__ __device__
    void swallow(Args&&... args) {}

    template<class... Events>
    __host__ __device__
    friend event when_all(cudaStream_t s, Events&... events)
    {
      // tell the stream to wait on all the events
      swallow(events.stream_wait(s)...);

      // invalidate the inputs
      swallow(events.destroy_event()...);

      // return a new event recorded on the stream
      return event(s);
    }
};


} // end detail
} // end cuda
} // end agency

