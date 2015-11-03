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
#include <agency/cuda/detail/unique_ptr.hpp>
#include <agency/cuda/detail/then_kernel.hpp>
#include <agency/cuda/detail/launch_kernel.hpp>
#include <agency/cuda/detail/workaround_unused_variable_warning.hpp>
#include <agency/cuda/detail/event.hpp>
#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/pointer.hpp>
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
      (std::is_void<T>::value && (sizeof...(Args) == 0))
    >
{};


template<class T>
struct state_requires_storage
  : std::integral_constant<
      bool,
      std::is_empty<T>::value || std::is_void<T>::value || agency::detail::is_empty_tuple<T>::value
   >
{};


// XXX should maybe call this asynchronous_state to match the nomenclature of the std
// XXX should try to collapse the implementation of this as much as possible between the two
template<class T,
         bool requires_storage = state_requires_storage<T>::value>
class future_state_impl
{
  public:
    using value_type = T;
    using pointer = value_type*;

    __host__ __device__
    future_state_impl() = default;

    template<class Arg1, class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Arg1,Args...>::value
             >::type
            >
    __host__ __device__
    future_state_impl(cudaStream_t s, Arg1&& ready_arg1, Args&&... ready_args)
      : data_(make_unique<T>(s, std::forward<Arg1>(ready_arg1), std::forward<Args>(ready_args)...))
    {}

    __host__ __device__
    future_state_impl(cudaStream_t s)
      : future_state_impl(s, T{})
    {}

    __host__ __device__
    future_state_impl(future_state_impl&& other) = default;

    __host__ __device__
    future_state_impl& operator=(future_state_impl&&) = default;

    __host__ __device__
    pointer data()
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

    __host__ __device__
    void swap(future_state_impl& other)
    {
      data_.swap(other.data_);
    }

  private:
    unique_ptr<T> data_;
};


// when a type is empty, we can create it on the fly upon dereference
template<class T>
struct empty_type_ptr : T
{
  using element_type = T;

  __host__ __device__
  T& operator*()
  {
    return *this;
  }

  __host__ __device__
  const T& operator*() const
  {
    return *this;
  }
};

template<>
struct empty_type_ptr<void> : unit_ptr {};


// zero storage optimization
template<class T>
class future_state_impl<T,true>
{
  public:
    using value_type = T;
    using pointer = empty_type_ptr<T>;

    __host__ __device__
    future_state_impl() : valid_(false) {}

    // constructs a valid state
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __host__ __device__
    future_state_impl(cudaStream_t, U&&) : valid_(true) {}

    // constructs a valid state
    __host__ __device__
    future_state_impl(cudaStream_t) : valid_(true) {}

    __host__ __device__
    future_state_impl(future_state_impl&& other) : valid_(other.valid_)
    {
      other.valid_ = false;
    }

    // 1. allow moves to void states (this simply discards the state)
    // 2. allow moves to empty types if the type can be constructed from an empty argument list
    template<class U,
             class T1 = T,
             class = typename std::enable_if<
               std::is_void<T1>::value ||
               (std::is_empty<T>::value && std::is_void<U>::value && std::is_constructible<T>::value)
             >::type>
    __host__ __device__
    future_state_impl(future_state_impl<U>&& other)
      : valid_(other.valid())
    {
      if(valid())
      {
        // invalidate the old state by calling .get() if it was valid when we received it
        other.get();
      }
    }

    __host__ __device__
    future_state_impl& operator=(future_state_impl&& other)
    {
      valid_ = other.valid_;
      other.valid_ = false;

      return *this;
    }

    __host__ __device__
    empty_type_ptr<T> data()
    {
      return empty_type_ptr<T>();
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
    void swap(future_state_impl& other)
    {
      bool other_valid_old = other.valid_;
      other.valid_ = valid_;
      valid_ = other_valid_old;
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


template<class T>
class future_state : public future_state_impl<T>
{
  public:
    using future_state_impl<T>::future_state_impl;
};
    

__host__ __device__
unit get_value_or_unit(future_state<void>&)
{
  return unit{};
}

template<class T>
__host__ __device__
T get_value_or_unit(future_state<T>& state)
{
  return state.get();
}

template<class T>
__host__ __device__
typename future_state<T>::pointer get_data_pointer(future_state<T>& state)
{
  return state.data();
}


template<class... Types>
__host__ __device__
static auto get_values_or_units(future_state<Types>&... states)
  -> decltype(
       agency::detail::make_tuple(get_value_or_unit(states)...)
     )
{
  return agency::detail::make_tuple(get_value_or_unit(states)...);
}

template<class... Types>
__host__ __device__
static auto get_data_pointers(future_state<Types>&... states)
  -> decltype(
       agency::detail::make_tuple(get_data_pointer(states)...)
     )
{
  return agency::detail::make_tuple(get_data_pointer(states)...);
}

struct get_values_or_units_from_tuple_functor
{
  template<class... Args>
  __host__ __device__
  auto operator()(Args&&... args) const
    -> decltype(
         get_values_or_units(std::forward<Args>(args)...)
       )
  {
    return get_values_or_units(std::forward<Args>(args)...);
  }
};


template<class... Types>
__host__ __device__
auto get_values_or_units_from_tuple(agency::detail::tuple<future_state<Types>...>& tuple)
  -> decltype(
       agency::detail::tuple_apply(get_values_or_units_from_tuple_functor{}, tuple)
     )
{
  return agency::detail::tuple_apply(get_values_or_units_from_tuple_functor{}, tuple);
}


struct get_data_pointers_from_tuple_functor
{
  template<class... Args>
  __host__ __device__
  auto operator()(Args&&... args) const -> decltype(get_data_pointers(std::forward<Args>(args)...))
  {
    return get_data_pointers(std::forward<Args>(args)...);
  }
};


template<class... Types>
__host__ __device__
auto get_data_pointers_from_tuple(agency::detail::tuple<future_state<Types>...>& tuple)
  -> decltype(
       agency::detail::tuple_apply(get_data_pointers_from_tuple_functor{}, tuple)
     )
{
  return agency::detail::tuple_apply(get_data_pointers_from_tuple_functor{}, tuple);
}


template<class... Types>
class future_state_tuple
{
  private:
    agency::detail::tuple<future_state<Types>...> state_tuple_;

    __host__ __device__
    decltype(get_values_or_units_from_tuple(state_tuple_))
      get_values_or_units_from_state_tuple()
    {
      return get_values_or_units_from_tuple(state_tuple_);
    }
    
    template<class T>
    struct is_not_unit : std::integral_constant<bool, !std::is_same<T,unit>::value> {};
    
    template<class Tuple>
    __host__ __device__
    static auto filter_non_unit(Tuple&& tuple)
      -> decltype(
           agency::detail::tuple_filter<is_not_unit>(std::forward<Tuple>(tuple))
         )
    {
      return agency::detail::tuple_filter<is_not_unit>(std::forward<Tuple>(tuple));
    }

    struct call_valid
    {
      template<class T>
      __AGENCY_ANNOTATION
      bool operator()(const future_state<T>& state) const
      {
        return state.valid();
      }
    };

  public:
    __host__ __device__
    future_state_tuple(future_state<Types>&... states)
      : state_tuple_(std::move(states)...)
    {}

    __host__ __device__
    auto get()
      -> decltype(
           agency::detail::unwrap_small_tuple(
             filter_non_unit(
               this->get_values_or_units_from_state_tuple()
             )
           )
         )
    {
      return agency::detail::unwrap_small_tuple(
        filter_non_unit(
          get_values_or_units_from_state_tuple()
        )
      );
    }

    using pointer = agency::detail::zip_pointer<typename future_state<Types>::pointer...>;

    __host__ __device__
    pointer data()
    {
      return get_data_pointers_from_tuple(state_tuple_);
    }

    __host__ __device__
    bool valid() const
    {
      return agency::detail::tuple_all_of(state_tuple_, call_valid{});
    }
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
    detail::event event_;
    detail::future_state<T> state_;

    // XXX stream_ should default to per-thread default stream
    static constexpr cudaStream_t default_stream{0};

  public:
    // XXX this should be private
    __host__ __device__
    future(cudaStream_t s) : future(s, 0) {}

    __host__ __device__
    future() : future(default_stream) {}

    __host__ __device__
    future(future&& other)
      : future()
    {
      future::swap(stream_, other.stream_);
      event_.swap(other.event_);
      state_.swap(other.state_);
    } // end future()

    __host__ __device__
    future &operator=(future&& other)
    {
      future::swap(stream_, other.stream_);
      future::swap(event_,  other.event_);
      future::swap(state_,  other.state_);
      return *this;
    } // end operator=()

    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 detail::future_state<T>,
                 detail::future_state<U>&&
               >::value
             >::type>
    __host__ __device__
    future(future<U>&& other)
      : stream_(),
        event_(),
        state_(std::move(other.state_))
    {
      future::swap(stream_, other.stream_);
      event_.swap(other.event_);
    } // end future()

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
    detail::event& event()
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
      detail::event ready_event(detail::event::construct_ready);

      return future(std::move(ready_event), std::forward<Args>(args)...);
    }

    // XXX this is only used by grid_executor::then_execute()
    __host__ __device__
    auto data() -> decltype(state_.data())
    {
      return state_.data();
    }

    template<class Function>
    __host__ __device__
    future<agency::detail::result_of_continuation_t<Function,future>>
      then(Function f)
    {
      // create state for the continuation's result
      using result_type = agency::detail::result_of_continuation_t<Function,future>;
      detail::future_state<result_type> result_state(stream());

      // get a pointer to the continuation's kernel
      using result_ptr_type = decltype(result_state.data());
      using arg_ptr_type = decltype(data());
      void (*kernel_ptr)(result_ptr_type, Function, arg_ptr_type) = detail::then_kernel<result_ptr_type, Function, arg_ptr_type>;
      detail::workaround_unused_variable_warning(kernel_ptr);

      // launch the continuation
      detail::event next_event = event().then(reinterpret_cast<void*>(kernel_ptr), dim3{1}, dim3{1}, 0, stream(), result_state.data(), f, data());

      // return the continuation's future
      return future<result_type>(stream(), std::move(next_event), std::move(result_state));
    }

  private:
    template<class U> friend class future;
    template<class Shape, class Index, class ThisIndexFunction> friend class agency::cuda::detail::basic_grid_executor;

    __host__ __device__
    future(cudaStream_t s, detail::event&& e, detail::future_state<T>&& state)
      : stream_(s), event_(std::move(e)), state_(std::move(state))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    future(cudaStream_t s, detail::event&& e, Args&&... ready_args)
      : future(s, std::move(e), detail::future_state<T>(s, std::forward<Args>(ready_args)...))
    {}

    template<class... Args,
             class = typename std::enable_if<
               detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __host__ __device__
    future(detail::event&& e, Args&&... ready_args)
      : future(default_stream, std::move(e), detail::future_state<T>(default_stream, std::forward<Args>(ready_args)...))
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
};


inline __host__ __device__
future<void> make_ready_future()
{
  return future<void>::make_ready();
} // end make_ready_future()


} // end namespace cuda
} // end namespace agency

