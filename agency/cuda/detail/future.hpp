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


namespace agency
{
namespace cuda
{
namespace detail
{


template<typename T> class future;


template<>
class future<void>
{
  public:

    __host__ __device__
    future()
      : stream_{0}, event_{0}
    {}

    __host__ __device__
    future(cudaEvent_t e)
      : stream_{0}, event_{e}
    {
    } // end future()

    __host__ __device__
    future(cudaStream_t s)
      : stream_{s}
    {
#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&event_, event_create_flags), "cudaEventCreateWithFlags in future ctor");
      detail::throw_on_error(cudaEventRecord(event_, stream_), "cudaEventRecord in future ctor");
#else
      detail::terminate_with_message("agency::cuda::detail::future ctor requires CUDART");
#endif // __cuda_lib_has_cudart
    } // end future()

    __host__ __device__
    future(future&& other)
      : stream_{0}, event_{0}
    {
      future::swap(stream_,      other.stream_);
      future::swap(event_,       other.event_);
    } // end future()

    __host__ __device__
    future &operator=(future&& other)
    {
      future::swap(stream_,      other.stream_);
      future::swap(event_,       other.event_);
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
          printf("CUDA error after cudaEventDestroy in future dtor: %s", cudaGetErrorString(e));
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
      detail::throw_on_error(cudaEventSynchronize(event_), "cudaEventSynchronize in future::wait");
#else
      // XXX need to capture the error as an exception and then throw it in .get()
      detail::throw_on_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize in future::wait");
#endif // __CUDA_ARCH__

#else
      detail::terminate_with_message("agency::cuda::detail::future::wait() requires CUDART");
#endif // __cuda_lib_has_cudart
    } // end wait()

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
    static future<void> make_ready()
    {
      cudaEvent_t ready_event = 0;

#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&ready_event, event_create_flags), "cudaEventCreateWithFlags in future<void>::make_ready_future");
#else
      detail::terminate_with_message("agency::cuda::detail::future::make_ready() requires CUDART");
#endif

      return future<void>{ready_event};
    }

  private:
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


inline __host__ __device__
future<void> make_ready_future()
{
  return future<void>::make_ready();
} // end make_ready_future()


} // end namespace detail
} // end namespace cuda
} // end namespace agency

