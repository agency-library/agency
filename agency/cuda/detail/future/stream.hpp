#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/device.hpp>
#include <agency/detail/utility.hpp>
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
    // creates a new stream associated with the given device
    inline __host__ __device__
    stream(device_id d)
      : device_(d)
    {
#if __cuda_lib_has_cudart
      // temporarily switch to our device while creating the stream
      scoped_device scope(device());

      detail::throw_on_error(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags in cuda::detail::stream ctor");
#else
      agency::detail::terminate_with_message("cuda::detail::stream ctor requires CUDART");
#endif
    }

    // creates a new stream associated with the current device
    // XXX do we actually want to make this depend on the current state of the CUDA runtime?
    // XXX it might be a better idea to associate the stream with device 0
    // XXX alternatively, maybe it would be better if a default-constructed stream was not associated with a device
    inline __host__ __device__
    stream()
      : stream(current_device())
    {}

    inline __host__ __device__
    stream(stream&& other)
      : device_(other.device())
    {
      s_ = other.release();
    }

    inline __host__ __device__
    ~stream()
    {
#if __cuda_lib_has_cudart
      if(valid())
      {
        // avoid propagating an exception but report the error if one exists
        detail::print_error_message_if(cudaStreamDestroy(release()), "cudaStreamDestroy in cuda::detail::stream dtor");
      }
#else
      agency::detail::terminate_with_message("cuda::detail::stream dtor requires CUDART");
#endif
    }

    inline __host__ __device__
    device_id device() const
    {
      return device_;
    }

    // returns the underlying cudaStream_t
    inline __host__ __device__
    cudaStream_t native_handle() const
    {
      return s_;
    }

    // releases ownership of the underlying cudaStream_t and invalidates this stream
    inline __host__ __device__
    cudaStream_t release()
    {
      cudaStream_t result = 0;
      agency::detail::adl_swap(result, s_);
      return result;
    }

    inline __host__ __device__
    bool valid() const
    {
      return native_handle() != 0;
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

    inline __host__ __device__
    void wait_on(cudaEvent_t e)
    {
#if __cuda_lib_has_cudart
      // make the next launch wait on the event
      throw_on_error(cudaStreamWaitEvent(native_handle(), e, 0), "cuda::detail::stream::wait_on(cudaEvent_t): cudaStreamWaitEvent()");
#else
      throw_on_error(cudaErrorNotSupported, "cuda::detail::stream::wait_on(cudaEvent_t): cudaStreamWaitEvent() requires CUDART");
#endif
    }

    inline __host__ __device__
    void wait_on(cudaStream_t s)
    {
#if __cuda_lib_has_cudart
      // record an event on s
      cudaEvent_t e;
      throw_on_error(cudaEventCreate(&e), "cuda::detail::stream::wait_on(cudaStream_t): cudaEventCreate()");
      throw_on_error(cudaEventRecord(e, s), "cuda::detail::stream::wait_on(cudaStream_t): cudaEventRecord()");

      // wait on the event
      wait_on(e);

      // destroy the event
      throw_on_error(cudaEventDestroy(e), "cuda::detail::stream::wait_on(cudaStream_t): cudaEventDestroy()");
#else
      throw_on_error(cudaErrorNotSupported, "cuda::detail::stream::wait_on(cudaStream_t): requires CUDART");
#endif
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

