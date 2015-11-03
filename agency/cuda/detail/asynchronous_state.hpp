#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/unique_ptr.hpp>
#include <type_traits>

namespace agency
{
namespace cuda
{
namespace detail
{


template<class T>
struct state_requires_storage
  : std::integral_constant<
      bool,
      std::is_empty<T>::value || std::is_void<T>::value || agency::detail::is_empty_tuple<T>::value
   >
{};


// XXX this is suffixed with _impl because of nvbug 1700337
//     eliminate the suffix when that bug is resolved
// XXX should try to collapse the implementation of this as much as possible between the two
template<class T,
         bool requires_storage = state_requires_storage<T>::value>
class asynchronous_state_impl
{
  public:
    using value_type = T;
    using pointer = value_type*;

    // constructs an invalid state
    __host__ __device__
    asynchronous_state_impl() = default;

    // constructs an immediately ready state
    template<class Arg1, class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Arg1,Args...>::value
             >::type
            >
    __host__ __device__
    asynchronous_state_impl(cudaStream_t s, Arg1&& ready_arg1, Args&&... ready_args)
      : data_(make_unique<T>(s, std::forward<Arg1>(ready_arg1), std::forward<Args>(ready_args)...))
    {}

    // constructs a not ready state
    __host__ __device__
    asynchronous_state_impl(cudaStream_t s)
      : asynchronous_state_impl(s, T{})
    {}

    __host__ __device__
    asynchronous_state_impl(asynchronous_state_impl&& other) = default;

    __host__ __device__
    asynchronous_state_impl& operator=(asynchronous_state_impl&&) = default;

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
    void swap(asynchronous_state_impl& other)
    {
      data_.swap(other.data_);
    }

  private:
    unique_ptr<T> data_;
};


// when a type is empty, we can create instances on the fly upon dereference
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
class asynchronous_state_impl<T,true>
{
  public:
    using value_type = T;
    using pointer = empty_type_ptr<T>;

    // constructs an invalid state
    __host__ __device__
    asynchronous_state_impl() : valid_(false) {}

    // constructs an immediately ready state
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U>::value
             >::type>
    __host__ __device__
    asynchronous_state_impl(cudaStream_t, U&&) : valid_(true) {}

    // constructs a not ready state
    __host__ __device__
    asynchronous_state_impl(cudaStream_t) : valid_(true) {}

    __host__ __device__
    asynchronous_state_impl(asynchronous_state_impl&& other) : valid_(other.valid_)
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
    asynchronous_state_impl(asynchronous_state_impl<U>&& other)
      : valid_(other.valid())
    {
      if(valid())
      {
        // invalidate the old state by calling .get() if it was valid when we received it
        other.get();
      }
    }

    __host__ __device__
    asynchronous_state_impl& operator=(asynchronous_state_impl&& other)
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
    void swap(asynchronous_state_impl& other)
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


// XXX this extra indirection is due to nvbug 1700337
//     eliminate this class when that bug is resolved
template<class T>
class asynchronous_state : public asynchronous_state_impl<T>
{
  public:
    using asynchronous_state_impl<T>::asynchronous_state_impl;
};

  
} // end detail
} // end cuda
} // end agency

