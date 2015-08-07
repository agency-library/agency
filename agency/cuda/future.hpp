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
#include <agency/cuda/detail/then_kernel.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class T, class... Args>
struct is_constructible_or_void
  : std::integral_constant<
      bool,
      std::is_constructible<T,Args...>::value ||
      std::is_void<T>::value && (sizeof...(Args) == 0)
    >
{};


// XXX should maybe call this asynchronous_state to match the nomenclature of the std
template<class T,
         bool requires_storage = std::is_empty<T>::value || std::is_void<T>::value>
class future_state
{
  public:
    __host__ __device__
    future_state() = default;

    template<class Arg1, class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Arg1,Args...>::value
             >::type
            >
    __host__ __device__
    future_state(cudaStream_t s, Arg1&& ready_arg1, Args&&... ready_args)
      : data_(make_unique<T>(s, std::forward<Arg1>(ready_arg1), std::forward<Args>(ready_args)...))
    {}

    __host__ __device__
    future_state(cudaStream_t s)
      : future_state(s, T{})
    {}

    __host__ __device__
    future_state(future_state&& other) : data_(std::move(other.data_)) {}

    __host__ __device__
    future_state& operator=(future_state&& other)
    {
      data_ = std::move(other.data_);
      return *this;
    }

    __host__ __device__
    T* data()
    {
      return data_.get();
    }

    __host__ __device__
    T get()
    {
      T result = std::move(*data_);

      data_.reset();

      return std::move(result);
    }

    __host__ __device__
    bool valid() const
    {
      return data_;
    }

    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args...>::value
             >::type>
    __host__ __device__
    void set_valid(cudaStream_t s, Args&&... ready_args)
    {
      data_ = make_unique<T>(s, std::forward<Args>(ready_args)...);
    }

  private:
    unique_ptr<T> data_;
};


// zero storage optimization
template<class T>
class future_state<T,true>
{
  public:
    __host__ __device__
    future_state() : valid_(false) {}

    // constructs a valid state
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __host__ __device__
    future_state(cudaStream_t, U&&) : valid_(true) {}

    // constructs a valid state
    __host__ __device__
    future_state(cudaStream_t) : valid_(true) {}

    __host__ __device__
    future_state(future_state&& other) : valid_(false)
    {
      valid_ = other.valid_;
      other.valid_ = false;
    }

    __host__ __device__
    future_state& operator=(future_state&& other)
    {
      valid_ = other.valid_;
      other.valid_ = false;

      return *this;
    }

    __host__ __device__
    // XXX WAR nvbug 1671566
    //std::nullptr_t data()
    my_nullptr_t data()
    {
      return nullptr;
    }

    __host__ __device__
    T get()
    {
      valid_ = false;

      return get_impl(std::is_void<T>());
    }

    __host__ __device__
    bool valid() const
    {
      return valid_;
    }

    __host__ __device__
    void set_valid(cudaStream_t)
    {
      valid_ = true;
    }

  private:
    __host__ __device__
    static T get_impl(std::false_type)
    {
      return T{};
    }

    __host__ __device__
    static T get_impl(std::true_type)
    {
      return;
    }

    bool valid_;
};


// declare this so future may befriend it
template<class Shape, class Index, class ThisIndexFunction>
class basic_grid_executor;


} // end detail


template<typename T>
class future
{
  private:
    cudaStream_t stream_;
    cudaEvent_t event_;
    detail::future_state<T> state_;

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
    T get()
    {
      wait();

      return state_.get();
    } // end get()

    __host__ __device__
    bool valid() const
    {
      return (event_ != 0) && state_.valid();
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

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    static future make_ready(Args&&... args)
    {
      cudaEvent_t ready_event = 0;

#if __cuda_lib_has_cudart
      detail::throw_on_error(cudaEventCreateWithFlags(&ready_event, event_create_flags), "cudaEventCreateWithFlags in agency::cuda::future<void>::make_ready");
#else
      detail::terminate_with_message("agency::cuda::future<void>::make_ready() requires CUDART");
#endif

      future result;
      result.set_valid(ready_event, std::forward<Args>(args)...);

      return result;
    }

    // XXX this is only used by grid_executor::then_execute()
    __host__ __device__
    auto data() -> decltype(state_.data())
    {
      return state_.data();
    }

    template<class Function>
    __host__ __device__
    future<typename std::result_of<Function()>::type>
      then(Function f)
    {
      using result_type = typename std::result_of<Function()>::type;

      detail::future_state<result_type> result_state(stream());

      using result_ptr_type = decltype(result_state.data());
      using arg_ptr_type = decltype(data());

      void (*kernel_ptr)(result_ptr_type, Function, arg_ptr_type) = detail::then_kernel<result_ptr_type, Function, arg_ptr_type>;
      detail::workaround_unused_variable_warning(kernel_ptr);

      cudaEvent_t next_event = detail::checked_launch_kernel_after_event_returning_next_event(reinterpret_cast<void*>(kernel_ptr), dim3{1}, dim3{1}, 0, stream(), event(), result_state.data(), f, data());

      return future<result_type>(stream(), next_event, std::move(result_state));
    }

    // XXX set_valid() should only be available to friends
    //     such as future<U> and grid_executor
    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    void set_valid(cudaEvent_t e, Args&&... args)
    {
      event_ = e;
      state_.set_valid(stream(), std::forward<Args>(args)...);
    }

  private:
    template<class U> friend class future;
    template<class Shape, class Index, class ThisIndexFunction> friend class agency::cuda::detail::basic_grid_executor;

    __host__ __device__
    future(cudaStream_t s, cudaEvent_t e, detail::future_state<T>&& state)
      : stream_(s), event_(e), state_(std::move(state))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    future(cudaStream_t s, cudaEvent_t e, Args&&... ready_args)
      : future(s, e, detail::future_state<T>(s, std::forward<Args>(ready_args)...))
    {}

    // implement swap to avoid depending on thrust::swap
    template<class U>
    __host__ __device__
    static void swap(U& a, U& b)
    {
      U tmp{a};
      a = b;
      b = tmp;
    }

    static const int event_create_flags = cudaEventDisableTiming;
};


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

