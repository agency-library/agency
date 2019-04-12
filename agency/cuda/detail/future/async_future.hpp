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
#include <agency/cuda/detail/future/event.hpp>
#include <agency/cuda/detail/future/asynchronous_state.hpp>
#include <agency/cuda/detail/future/continuation.hpp>
#include <agency/cuda/device.hpp>
#include <agency/detail/unit.hpp>
#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/tuple.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace cuda
{
namespace detail
{


template<class U>
using element_type_is_not_unit = std::integral_constant<
  bool,
  !std::is_same<typename std::pointer_traits<U>::element_type, agency::detail::unit>::value
>;


} // end detail


// forward declarations for async_future<T>'s benefit
template<class T>
class shared_future;

template<class T>
class future;

template<class T>
class async_future;

namespace detail
{

template<class T>
__host__ __device__
async_future<T> make_async_future(event&& e, asynchronous_state<T>&& state);

template<class T>
__host__ __device__
event& async_future_event(async_future<T>& future);

template<class T>
__host__ __device__
const event& async_future_event(const async_future<T>& future);

template<class T>
__host__ __device__
asynchronous_state<T>& async_future_state(async_future<T>& future);

} // end detail


template<class T>
class async_future
{
  private:
    detail::event event_;
    detail::asynchronous_state<T> state_;

  public:
    async_future() = default;

    __host__ __device__
    async_future(async_future&& other)
      : async_future()
    {
      event_.swap(other.event_);
      state_.swap(other.state_);
    } // end async_future()

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::asynchronous_state<T>,
                 detail::asynchronous_state<U>&&
               >::value
             >::type>
    __host__ __device__
    async_future(async_future<U>&& other)
      : event_(),
        state_(std::move(other.state_))
    {
      event_.swap(other.event_);
    } // end async_future()

    __host__ __device__
    ~async_future()
    {
      if(state_.valid())
      {
        // in order to avoid blocking on our state's destruction,
        // schedule it to be destroyed sometime in the future
        detail::async_invalidate_and_destroy(state_);
      }
    } // end async_future()

    __host__ __device__
    async_future &operator=(async_future&& other)
    {
      event_.swap(other.event_);
      state_.swap(other.state_);
      return *this;
    } // end operator=()

    __host__ __device__
    void wait() const
    {
      event_.wait();
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
      return event_.valid() && state_.valid();
    } // end valid()

    __host__ __device__
    bool is_ready() const
    {
      return event_.is_ready();
    } // end is_ready()

    template<class... Args,
             class = typename std::enable_if<
               agency::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    static async_future make_ready(Args&&... args)
    {
      detail::event ready_event(detail::event::construct_ready);

      return async_future(std::move(ready_event), std::forward<Args>(args)...);
    }

    template<class Function>
    __host__ __device__
    async_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        async_future
      >
    >
      then(Function f)
    {
      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type,async_future>;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready, cuda::allocator<result_type>());

      // tuple up f's input state
      auto unfiltered_pointer_tuple = agency::make_tuple(state().data());

      // filter void states
      auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);

      // make a function implementing the continuation
      auto continuation = detail::make_continuation(std::forward<Function>(f), result_state.data(), pointer_tuple);

      // launch the continuation
      detail::event next_event = event().then_and_invalidate(std::move(continuation), dim3{1}, dim3{1}, 0);

      // schedule our state for destruction when the next event is complete
      detail::invalidate_and_destroy_when(state_, next_event);

      // return the continuation's future
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    // XXX the implementation of async_future<T>::share() is in shared_future.hpp
    //     because it needs access to shared_future<T>'s class definition
    shared_future<T> share();

    // implement swap to avoid depending on thrust::swap
    template<class U>
    __host__ __device__
    static void swap(U& a, U& b)
    {
      U tmp{a};
      a = b;
      b = tmp;
    }

  private:
    template<class U> friend class async_future;
    template<class U> friend class agency::cuda::future;

    // friend detail::make_async_future() to give it access to the private constructor
    template<class U> friend async_future<U> detail::make_async_future(detail::event&& e, detail::asynchronous_state<U>&& state);

    // friend detail::async_future_event() to give it access to .event()
    template<class U>
    friend detail::event& detail::async_future_event(async_future<U>& future);

    template<class U>
    friend const detail::event& detail::async_future_event(const async_future<U>& future);

    // friend detail::async_future_state() to give it access to .state()
    template<class U>
    friend detail::asynchronous_state<U>& detail::async_future_state(async_future<U>& future);

    template<class... Types>
    friend __host__ __device__
    async_future<
      agency::detail::when_all_result_t<
        async_future<Types>...
      >
    >
    when_all(async_future<Types>&... futures);


    __host__ __device__
    detail::event& event()
    {
      return event_;
    } // end event()

    __host__ __device__
    const detail::event& event() const
    {
      return event_;
    } // end event()

    __host__ __device__
    detail::asynchronous_state<T>& state()
    {
      return state_;
    }

    template<class U>
    __host__ __device__
    static U get_ref_impl(agency::detail::empty_type_ptr<U> ptr)
    {
      return *ptr;
    }

    template<class U>
    __host__ __device__
    static U& get_ref_impl(U* ptr)
    {
      return *ptr;
    }

    // XXX get_ref() should be a member of asynchronous_state
    __host__ __device__
    auto get_ref() ->
      decltype(get_ref_impl(this->state().data()))
    {
      wait();
      return get_ref_impl(state().data());
    }

    // this version of then() leaves the future in a valid state
    // it's used by cuda::future
    template<class Function>
    __host__ __device__
    async_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        async_future
      >
    >
      then_and_leave_valid(Function f)
    {
      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<typename std::decay<Function>::type,async_future>;
      detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready);

      // tuple up f's input state
      auto unfiltered_pointer_tuple = agency::make_tuple(state().data());

      // filter void states
      auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);
      
      // make a function implementing the continuation
      auto continuation = detail::make_continuation(std::forward<Function>(f), result_state.data(), pointer_tuple);

      // launch the continuation
      detail::event next_event = event().then(std::move(continuation), dim3{1}, dim3{1}, 0);

      // return the continuation's future
      return async_future<result_type>(std::move(next_event), std::move(result_state));
    }

    __host__ __device__
    async_future(detail::event&& e, detail::asynchronous_state<T>&& state)
      : event_(std::move(e)), state_(std::move(state))
    {}

    // XXX this constructor should take an allocator argument and forward it to the asynchronous_state constructor
    template<class... Args,
             class = typename std::enable_if<
               agency::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    async_future(detail::event&& e, Args&&... ready_args)
      : async_future(std::move(e), detail::asynchronous_state<T>(agency::detail::construct_ready, cuda::allocator<T>(), std::forward<Args>(ready_args)...))
    {}
};


inline __host__ __device__
async_future<void> make_ready_async_future()
{
  return async_future<void>::make_ready();
}


template<class T>
__host__ __device__
async_future<T> make_ready_async_future(T&& val)
{
  return async_future<T>::make_ready(std::forward<T>(val));
}


namespace detail
{


template<class T>
__host__ __device__
async_future<T> make_async_future(event&& e, asynchronous_state<T>&& state)
{
  return async_future<T>(std::move(e), std::move(state));
}


template<class T>
__host__ __device__
event& async_future_event(async_future<T>& future)
{
  return future.event();
}


template<class T>
__host__ __device__
const event& async_future_event(const async_future<T>& future)
{
  return future.event();
}


template<class T>
__host__ __device__
asynchronous_state<T>& async_future_state(async_future<T>& future)
{
  return future.state();
}


template<class T>
__host__ __device__
agency::tuple<event, asynchronous_state<T>> invalidate_async_future(async_future<T>& future)
{
  // we invalidate the future by moving its event and state elsewhere

  event& e = detail::async_future_event(future);
  asynchronous_state<T>& state = detail::async_future_state(future);

  return agency::tuple<event, asynchronous_state<T>>(std::move(e), std::move(state));
}


template<class Result>
struct when_all_functor
{
  template<class... Args>
  __host__ __device__
  Result operator()(Args&... args)
  {
    return Result(std::move(args)...);
  }
};


} // end detail


template<class... Types>
__host__ __device__
async_future<
  agency::detail::when_all_result_t<
    async_future<Types>...
  >
>
when_all(async_future<Types>&... futures)
{
  // join the events
  detail::event when_all_ready = detail::when_all_events_are_ready(futures.event()...);

  using result_type = agency::detail::when_all_result_t<
    async_future<Types>...
  >;

  detail::asynchronous_state<result_type> result_state(agency::detail::construct_not_ready, cuda::allocator<result_type>());

  // tuple up the input states
  auto unfiltered_pointer_tuple = agency::make_tuple(futures.state().data()...);

  // filter void states
  auto pointer_tuple = agency::detail::tuple_filter<detail::element_type_is_not_unit>(unfiltered_pointer_tuple);

  // make a function implementing the continuation
  auto continuation = detail::make_continuation(detail::when_all_functor<result_type>{}, result_state.data(), pointer_tuple);

  // launch the continuation
  detail::event next_event = when_all_ready.then_and_invalidate(continuation, dim3{1}, dim3{1}, 0);

  // return the continuation's future
  return async_future<result_type>(std::move(next_event), std::move(result_state));
}


template<class T>
__host__ __device__
async_future<T> when_all(async_future<T>& future)
{
  return std::move(future);
}


namespace experimental
{


// returns a new cudaStream_t which depends on the given async_future
// it is the caller's responsibility to destroy the cudaStream_t with cudaStreamDestroy()
template<class T>
__host__ __device__
cudaStream_t make_dependent_stream(const async_future<T>& future)
{
  return cuda::detail::async_future_event(future).make_dependent_stream().release();
}


// returns an async_future<void> whose readiness depends on the completion of the given event
__host__ __device__
inline async_future<void> make_async_future(cudaEvent_t e)
{
  // create a new stream on device 0
  device_id device(0);
  cuda::detail::stream s{device};
  assert(s.native_handle() != 0);

  // tell the stream to wait on e
  s.wait_on(e);

  // create a new event
  cuda::detail::event event(std::move(s));
  assert(event.valid());

  // create a new, not ready asynchronous_state
  cuda::detail::asynchronous_state<void> state(agency::detail::construct_not_ready, cuda::allocator<void>());
  assert(state.valid());

  return make_async_future(std::move(event), std::move(state));
}


namespace detail
{


// returns an async_future<T> whose readiness depends on the completion of the given event
template<class T>
async_future<T> make_async_future(cudaEvent_t e, agency::cuda::detail::asynchronous_state<T>&& state)
{
  // create a new stream on device 0
  device_id device(0);
  cuda::detail::stream s{device};
  assert(s.native_handle() != 0);

  // tell the stream to wait on e
  s.wait_on(e);

  // create a new event
  cuda::detail::event event(std::move(s));
  assert(event.valid());

  return make_async_future(std::move(event), std::move(state));
}


} // end detail


// returns an async_future<T> whose readiness depends on the completion of the given event
template<class T, class Allocator>
async_future<T> make_async_future(cudaEvent_t e, T* ptr, const Allocator& allocator)
{
  static_assert(agency::detail::is_allocator<Allocator>::value, "allocator parameter is not an Allocator.");

  // create a new, not ready asynchronous state
  cuda::detail::asynchronous_state<T> state(agency::detail::construct_not_ready, ptr, allocator);
  assert(state.valid());

  return detail::make_async_future(e, std::move(state));
}


// returns an async_future<void> whose readiness depends on the completion of all events previously recorded on the given stream
__host__ __device__
inline async_future<void> make_async_future(cudaStream_t s)
{
  // create an event corresponding to the completion of everything submitted so far to s
  cuda::detail::event e = cuda::detail::when_all_events_are_ready(s);

  return experimental::make_async_future(e.native_handle());
}


// returns an async_future<T> whose readiness depends on the completion of all events previously recorded on the given stream
template<class T, class Allocator>
async_future<T> make_async_future(cudaStream_t s, T* ptr, const Allocator& allocator)
{
  static_assert(agency::detail::is_allocator<Allocator>::value, "allocator parameter is not an Allocator.");

  // create an event corresponding to the completion of everything submitted so far to s
  cuda::detail::event e = cuda::detail::when_all_events_are_ready(s);

  return experimental::make_async_future(e.native_handle(), ptr, allocator);
}


} // end experimental
} // end namespace cuda
} // end namespace agency

