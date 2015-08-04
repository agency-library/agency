/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/feature_test.hpp>
#include <agency/cuda/detail/terminate.hpp>
#include <agency/cuda/detail/unique_ptr.hpp>
#include <utility>


namespace agency
{
namespace cuda
{


template<typename T> class future;


template<>
class future<void>
{
  public:
    // XXX this should be private
    __host__ __device__
    future(cudaStream_t s) : future(s, 0) {}

    // XXX stream_ should default to per-thread default stream
    __host__ __device__
    future() : future(0) {}

    __host__ __device__
    future(future&& other)
      : future()
    {
      future::swap(stream_, other.stream_);
      future::swap(event_,  other.event_);
    } // end future()

    __host__ __device__
    future &operator=(future&& other)
    {
      future::swap(stream_, other.stream_);
      future::swap(event_,  other.event_);
      return *this;
    } // end operator=()

    __host__ __device__
    ~future()
    {
      if(valid())
      {
#if __cuda_lib_has_cudart
        // swallow errors
        cudaError_t e = cudaEventDestroy(event_);

#if __cuda_lib_has_printf
        if(e)
        {
          printf("CUDA error after cudaEventDestroy in agency::cuda::future<void> dtor: %s", cudaGetErrorString(e));
        } // end if
#endif // __cuda_lib_has_printf
#endif // __cuda_lib_has_cudart
      } // end if
    } // end ~future()

    __host__ __device__
    void wait() const
    {
      // XXX should probably check for valid() here

#if __cuda_lib_has_cudart

#ifndef __CUDA_ARCH__
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::throw_on_error(cudaEventSynchronize(event_), "cudaEventSynchronize in agency::cuda<void>::future::wait");
#else
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::throw_on_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize in agency::cuda<void>::future::wait");
#endif // __CUDA_ARCH__

#else
      detail::terminate_with_message("agency::cuda::future<void>::wait() requires CUDART");
#endif // __cuda_lib_has_cudart
    } // end wait()

    __host__ __device__
    void get() const
    {
      wait();
    } // end get()

    __host__ __device__
    bool valid() const
    {
      return event_ != 0;
    } // end valid()

    __host__ __device__
    cudaEvent_t event() const
    {
      return event_;
    } // end event()

    __host__ __device__
    cudaStream_t stream() const
    {
      return stream_;
    } // end stream()

    __host__ __device__
    static future<void> make_ready()
    {
      cudaEvent_t ready_event = 0;

#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&ready_event, event_create_flags), "cudaEventCreateWithFlags in agency::cuda::future<void>::make_ready");
#else
      detail::terminate_with_message("agency::cuda::future<void>::make_ready() requires CUDART");
#endif

      future<void> result;
      result.set_valid(ready_event);

      return result;
    }

    // XXX this is only used by grid_executor::then_execute()
    __host__ __device__
    std::nullptr_t ptr()
    {
      return nullptr;
    }

    // XXX set_valid() should only be available to friends
    //     such as future<T> and grid_executor
    __host__ __device__
    void set_valid(cudaEvent_t e)
    {
      event_ = e;
    }

  private:
    __host__ __device__
    future(cudaStream_t s, cudaEvent_t e) : stream_(s), event_(e) {}

    // implement swap to avoid depending on thrust::swap
    template<class T>
    __host__ __device__
    static void swap(T& a, T& b)
    {
      T tmp{a};
      a = b;
      b = tmp;
    }

    static const int event_create_flags = cudaEventDisableTiming;

    cudaStream_t stream_;
    cudaEvent_t event_;
}; // end future<void>


template<class T>
class future
{
  public:
    __host__ __device__
    future()
      : event_()
    {}

    // XXX this should be private
    // XXX this constructor should not even exist
    //     the ready event should be created in event_'s initializer
    template<class U>
    __host__ __device__
    future(U&& value, future<void>& e)
      : event_(std::move(e)),
        value_(detail::make_unique<T>(event_.stream(), std::forward<U>(value)))
    {
    } // end future()

    __host__ __device__
    future(cudaStream_t s)
      : event_(s)
    {
    } // end future()

    __host__ __device__
    future(future&& other)
      : event_(std::move(other.event_)),
        value_(std::move(other.value_))
    {
    } // end future()

    __host__ __device__
    future &operator=(future&& other)
    {
      event_ = std::move(other.event_);
      value_ = std::move(other.value_);
      return *this;
    } // end operator=()

    __host__ __device__
    void wait() const
    {
      event_.wait();
    } // end wait()

    __host__ __device__
    T get() const
    {
      wait();

      return *value_;
    } // end get()

    __host__ __device__
    future<void> discard_value()
    {
      return std::move(event_);
    } // end discard_value()

    __host__ __device__
    bool valid() const
    {
      return event_.valid();
    } // end valid()

    // XXX only used by grid_executor
    //     think of a better way to expose this
    // XXX the existence of future_cast makes this superfluous i think
    __host__ __device__
    future<void>& void_future()
    {
      return event_;
    } // end void_future()

    template<class U>
    __host__ __device__
    static future<T> make_ready(U&& value)
    {
      auto event = future<void>::make_ready();

      return future<T>{std::forward<U>(value), event};
    }

    // XXX this is only used by grid_executor::then_execute()
    __host__ __device__
    T* ptr()
    {
      return value_.get();
    }

    // XXX set_valid() should only be available to friends
    //     such as future<T> and grid_executor
    __host__ __device__
    void set_valid(cudaEvent_t event)
    {
      event_.set_valid(event);
    }

  private:
    future<void> event_;
    detail::unique_ptr<T> value_;
}; // end future<T>


inline __host__ __device__
future<void> make_ready_future()
{
  return future<void>::make_ready();
} // end make_ready_future()


template<class T>
inline __host__ __device__
future<typename std::decay<T>::type> make_ready_future(T&& value)
{
  return future<typename std::decay<T>::type>::make_ready(std::forward<T>(value));
} // end make_ready_future()


} // end namespace cuda
} // end namespace agency

