#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/kernel.hpp>
#include <agency/cuda/detail/future/stream.hpp>
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
    static constexpr construct_ready_t construct_ready{};

  private:
    static constexpr int event_create_flags = cudaEventDisableTiming;

    __host__ __device__
    event(construct_ready_t, detail::stream&& s)
      : stream_(std::move(s))
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&e_, event_create_flags), "cudaEventCreateWithFlags in cuda::detail::event ctor");
#else
      detail::terminate_with_message("cuda::detail::event ctor requires CUDART");
#endif
    }

    // constructs a new event recorded on the given stream
    __host__ __device__
    event(detail::stream&& s) : event(construct_ready, std::move(s))
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventRecord(e_, stream().native_handle()), "cudaEventRecord in cuda::detail::event ctor");
#else
      detail::terminate_with_message("cuda::detail::event ctor requires CUDART");
#endif
    }

  public:
    // creates an invalid event
    __host__ __device__
    event() : stream_(), e_{0}
    {}

    __host__ __device__
    event(construct_ready_t)
      : event(construct_ready, detail::stream())
    {}

    __host__ __device__
    event(const event&) = delete;

    __host__ __device__
    event(event&& other)
      : stream_(std::move(other.stream())),
        e_(other.e_)
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
      stream().swap(other.stream());

      cudaEvent_t tmp = e_;
      e_ = other.e_;
      other.e_ = tmp;
    }

    template<class Function, class... Args>
    __host__ __device__
    static void *then_kernel()
    {
      return reinterpret_cast<void*>(&cuda_kernel<Function,Args...>);
    }

    template<class Function, class... Args>
    __host__ __device__
    static void *then_kernel(const Function&, const Args&...)
    {
      return then_kernel<Function,Args...>();
    }

    template<class Function, class... Args>
    __host__ __device__
    event then(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const Args&... args)
    {
      // make the stream wait before launching
      stream_wait();

      // get the address of the kernel
      auto kernel = then_kernel(f,args...);

      // launch the kernel
      detail::checked_launch_kernel(kernel, grid_dim, block_dim, shared_memory_size, stream().native_handle(), f, args...);

      return event(std::move(stream()));
    }

    template<class Function, class... Args>
    __host__ __device__
    static void *then_on_kernel()
    {
      return reinterpret_cast<void*>(&cuda_kernel<Function,Args...>);
    }

    template<class Function, class... Args>
    __host__ __device__
    static void *then_on_kernel(const Function&, const Args&...)
    {
      return then_on_kernel<Function,Args...>();
    }

    template<class Function, class... Args>
    __host__ __device__
    event then_on(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const gpu_id& gpu, const Args&... args)
    {
      // make the stream wait before launching
      stream_wait();

      // get the address of the kernel
      auto kernel = then_on_kernel(f,args...);

      // if gpu differs from this event's stream's gpu, we need to create a new one for the launch
      // otherwise, just reuse this event's stream
      detail::stream new_stream = (gpu == stream().gpu()) ? std::move(stream()) : detail::stream(gpu);

      detail::checked_launch_kernel_on_device(kernel, grid_dim, block_dim, shared_memory_size, new_stream.native_handle(), gpu.native_handle(), f, args...);

      return event(std::move(new_stream));
    }

  private:
    stream stream_;
    cudaEvent_t e_;

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

    __host__ __device__
    stream& stream()
    {
      return stream_;
    }

    // makes the given stream wait on this event and invalidates this event
    // this function returns 0 so that we can pass it as an argument to swallow(...)
    __host__ __device__
    int stream_wait(const detail::stream& s)
    {
#if __cuda_lib_has_cudart
      // make the next launch wait on the event
      throw_on_error(cudaStreamWaitEvent(s.native_handle(), e_, 0), "cuda::detail::event::stream_wait(): cudaStreamWaitEvent()");
#else
      throw_on_error(cudaErrorNotSupported, "cuda::detail::event::stream_wait(): cudaStreamWaitEvent() requires CUDART");
#endif // __cuda_lib_has_cudart

      // this operation invalidates this event
      destroy_event();

      return 0;
    }

    __host__ __device__
    int stream_wait()
    {
      return stream_wait(stream());
    }

    template<class... Args>
    inline __host__ __device__
    static void swallow(Args&&... args) {}

    template<class... Events>
    __host__ __device__
    friend event when_all_events_are_ready(const gpu_id& gpu, Events&... events);

    template<class... Events>
    __host__ __device__
    friend event when_all_events_are_ready(Events&... events);
};


template<class... Events>
__host__ __device__
event when_all_events_are_ready(const gpu_id& gpu, Events&... events)
{
  detail::stream s{gpu};

  // tell the stream to wait on all the events
  event::swallow(events.stream_wait(s)...);

  // return a new event recorded on the stream
  return event(std::move(s));
}


template<class... Events>
__host__ __device__
event when_all_events_are_ready(Events&... events)
{
  // just use the current gpu
  // XXX we might prefer the gpu associated with the first event
  return agency::cuda::detail::when_all_events_are_ready(current_gpu(), events...);
}


} // end detail
} // end cuda
} // end agency

