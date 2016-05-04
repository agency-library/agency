#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/kernel.hpp>
#include <agency/cuda/detail/future/stream.hpp>
#include <agency/cuda/device.hpp>

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
      // switch to the stream's device when creating the event
      scoped_current_device scope(stream().device());

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
      : event()
    {
      swap(other);
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
    event& operator=(event&& other)
    {
      swap(other);
      return *this;
    }

    __host__ __device__
    bool valid() const
    {
      return e_ != 0;
    }

    __host__ __device__
    bool is_ready() const
    {
      if(valid())
      {
#if __cuda_lib_has_cudart
        cudaError_t result = cudaEventQuery(e_);

        if(result != cudaErrorNotReady && result != cudaSuccess)
        {
          detail::throw_on_error(result, "cudaEventQuery in cuda::detail::event::is_ready");
        }

        return result == cudaSuccess;
#else
        detail::terminate_with_message("cuda::detail::event::is_ready requires CUDART");
#endif
      }

      return false;
    }

    __host__ __device__
    cudaError_t wait_and_return_cuda_error() const
    {
#if __cuda_lib_has_cudart
#  ifndef __CUDA_ARCH__
      return cudaEventSynchronize(e_);
#  else
      return cudaDeviceSynchronize();
#  endif // __CUDA_ARCH__
#else
      return cudaErrorNotSupported;
#endif // __cuda_lib_has_cudart
    }

    __host__ __device__
    void wait() const
    {
      // XXX should probably check for valid() here

      detail::throw_on_error(wait_and_return_cuda_error(), "wait_and_return_cuda_error in cuda::detail::event::wait");
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

    // this form of then() leaves this event in a valid state afterwards
    template<class Function, class... Args>
    __host__ __device__
    event then(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const Args&... args)
    {
      return then_on(f, grid_dim, block_dim, shared_memory_size, stream().device(), args...);
    }

    // this form of then() leaves this event in an invalid state afterwards
    template<class Function, class... Args>
    __host__ __device__
    event then_and_invalidate(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const Args&... args)
    {
      // make the stream wait on this event before further launches
      stream_wait_and_invalidate();

      // get the address of the kernel
      auto kernel = then_kernel(f,args...);

      // launch the kernel on this event's stream
      detail::checked_launch_kernel(kernel, grid_dim, block_dim, shared_memory_size, stream().native_handle(), f, args...);

      // return a new event
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

    // this form of then_on() leaves this event in an invalid state afterwards
    template<class Function, class... Args>
    __host__ __device__
    event then_on_and_invalidate(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const device_id& device, const Args&... args)
    {
      // if device differs from this event's stream's device, we need to create a new one for the launch
      // otherwise, just reuse this event's stream
      detail::stream new_stream = (device == stream().device()) ? std::move(stream()) : detail::stream(device);

      // make the new stream wait on this event
      stream_wait(new_stream, *this);

      // get the address of the kernel
      auto kernel = then_on_kernel(f,args...);

      // launch the kernel on the new stream
      detail::checked_launch_kernel_on_device(kernel, grid_dim, block_dim, shared_memory_size, new_stream.native_handle(), device.native_handle(), f, args...);

      // invalidate this event
      *this = event();

      // return a new event
      return event(std::move(new_stream));
    }

    // this form of then_on() leaves this event in a valid state afterwards
    template<class Function, class... Args>
    __host__ __device__
    event then_on(Function f, dim3 grid_dim, dim3 block_dim, int shared_memory_size, const device_id& device, const Args&... args)
    {
      // create a stream for the kernel on the given device
      detail::stream new_stream(device);

      // make the new stream wait on this event
      stream_wait(new_stream, *this);

      // get the address of the kernel
      auto kernel = then_on_kernel(f,args...);

      // launch the kernel on the new stream
      detail::checked_launch_kernel_on_device(kernel, grid_dim, block_dim, shared_memory_size, new_stream.native_handle(), device.native_handle(), f, args...);

      // return a new event
      return event(std::move(new_stream));
    }

    // this form of then() leaves this event in a valid state afterwards
    // XXX might want to see if we can receive f by forwarding reference
    template<class Function>
    __host__ __device__
    event then(Function f)
    {
#ifndef __CUDA_ARCH__
      // if on host, use a stream callback
      // if on device, use then_on()

      // create a stream for the callback on the current device
      detail::stream new_stream(stream().device());

      // make the new stream wait on this event
      stream_wait(new_stream, *this);

      // launch f on the new stream
      new_stream.add_callback(f);

      // return a new event
      return event(std::move(new_stream));
#else
      detail::terminate_with_message("cuda::detail::event::then(): unimplemented function called.");
      return event();
      // launch a single-thread kernel
      //return then_on([=](uint3, uint3){ f(); }, dim3{1}, dim3{1}, 0, stream().device());
#endif
    }

    __host__ __device__
    cudaEvent_t native_handle() const
    {
      return e_;
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

      detail::print_error_message_if(error, "CUDA error after cudaEventDestroy in cuda::detail::event::destroy_event");
#endif // __cuda_lib_has_cudart

      return 0;
    }

    __host__ __device__
    stream& stream()
    {
      return stream_;
    }

    // makes the given stream wait on the given event
    // XXX this should be moved into the implementation of detail::stream
    __host__ __device__
    static void stream_wait(const detail::stream& s, const detail::event& e)
    {
#if __cuda_lib_has_cudart
      // make the next launch wait on the event
      throw_on_error(cudaStreamWaitEvent(s.native_handle(), e.e_, 0), "cuda::detail::event::stream_wait(): cudaStreamWaitEvent()");
#else
      throw_on_error(cudaErrorNotSupported, "cuda::detail::event::stream_wait(): cudaStreamWaitEvent() requires CUDART");
#endif // __cuda_lib_has_cudart
    }

    // makes the given stream wait on this event
    __host__ __device__
    int stream_wait(const detail::stream& s) const
    {
      stream_wait(s, *this);

      return 0;
    }

    // makes the given stream wait on this event and invalidates this event
    // this function returns 0 so that we can pass it as an argument to swallow(...)
    __host__ __device__
    int stream_wait_and_invalidate(const detail::stream& s)
    {
      stream_wait(s);

      // this operation invalidates this event
      destroy_event();

      return 0;
    }

    // makes this event's stream wait on this event and invalidates this event
    __host__ __device__
    int stream_wait_and_invalidate()
    {
      return stream_wait_and_invalidate(stream());
    }

    template<class... Args>
    inline __host__ __device__
    static void swallow(Args&&... args) {}

    template<class... Events>
    __host__ __device__
    friend event when_all_events_are_ready(const device_id& device, Events&... events);

    template<class... Events>
    __host__ __device__
    friend event when_all_events_are_ready(Events&... events);
};


inline __host__ __device__
event make_ready_event()
{
  return event(event::construct_ready);
}


template<class... Events>
__host__ __device__
event when_all_events_are_ready(const device_id& device, Events&... events)
{
  detail::stream s{device};

  // tell the stream to wait on all the events
  event::swallow(events.stream_wait_and_invalidate(s)...);

  // return a new event recorded on the stream
  return event(std::move(s));
}


template<class... Events>
__host__ __device__
event when_all_events_are_ready(Events&... events)
{
  // just use the current device
  // XXX we might prefer the device associated with the first event
  return agency::cuda::detail::when_all_events_are_ready(current_device(), events...);
}


// a blocking_event is an event whose destructor calls .wait() when the blocking_event is valid
class blocking_event : public event
{
  public:
    inline __host__ __device__
    blocking_event(blocking_event&& other)
      : event(std::move(other))
    {}

    inline __host__ __device__
    blocking_event(event&& other)
      : event(std::move(other))
    {}

    inline __host__ __device__
    ~blocking_event()
    {
      if(valid())
      {
        // since we're in a destructor, let's avoid
        // propagating exceptions out of a destructor
        detail::print_error_message_if(wait_and_return_cuda_error(), "wait_and_return_cuda_error in cuda::detail::blocking_event() dtor");
      }
    }
};


} // end detail
} // end cuda
} // end agency

