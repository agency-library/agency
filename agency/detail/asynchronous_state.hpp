#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/unique_ptr.hpp>
#include <agency/detail/tuple.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


struct unit_ptr : unit
{
  using element_type = unit;

  __AGENCY_ANNOTATION
  unit& operator*()
  {
    return *this;
  }

  __AGENCY_ANNOTATION
  const unit& operator*() const
  {
    return *this;
  }
};


template<class T>
struct state_requires_storage
  : std::integral_constant<
      bool,
      std::is_empty<T>::value || std::is_void<T>::value || agency::detail::is_empty_tuple<T>::value
   >
{};

struct construct_ready_t {};
struct construct_not_ready_t {};

constexpr static construct_ready_t     construct_ready{};
constexpr static construct_not_ready_t construct_not_ready{};


// XXX this is suffixed with _impl because of nvbug 1700337
//     eliminate the suffix when that bug is resolved
// XXX should try to collapse the implementation of this as much as possible between the two
template<class T,
         class Alloc = std::allocator<T>,
         bool requires_storage = state_requires_storage<T>::value>
class asynchronous_state_impl
{
  public:
    using value_type = T;
    using pointer = value_type*;
    using storage_type = unique_ptr<T,deleter<Alloc>>;

    // constructs an invalid state
    __AGENCY_ANNOTATION
    asynchronous_state_impl() = default;

    // constructs an immediately ready state
    __agency_exec_check_disable__
    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args...>::value
             >::type
            >
    __AGENCY_ANNOTATION
    asynchronous_state_impl(construct_ready_t, Args&&... ready_args)
      : storage_(allocate_unique<T>(Alloc(), std::forward<Args>(ready_args)...))
    {}

    // constructs a not ready state
    // XXX we should avoid creating an object here
    //     instead, we should just create it uninitialized
    // XXX the destructor should check whether the state requires destruction
    __AGENCY_ANNOTATION
    asynchronous_state_impl(construct_not_ready_t)
      : asynchronous_state_impl(construct_ready, T{})
    {}

    __AGENCY_ANNOTATION
    asynchronous_state_impl(asynchronous_state_impl&& other) = default;

    template<class OtherT,
             class OtherAlloc,
             class = typename std::enable_if<
               std::is_constructible<storage_type, typename asynchronous_state_impl<OtherT,OtherAlloc>::storage_type&&>::value
             >::type
            >
    __AGENCY_ANNOTATION
    asynchronous_state_impl(asynchronous_state_impl<OtherT,OtherAlloc>&& other)
      : storage_(std::move(other.storage_))
    {}

    __AGENCY_ANNOTATION
    asynchronous_state_impl& operator=(asynchronous_state_impl&&) = default;

    __AGENCY_ANNOTATION
    pointer data() const
    {
      return storage_.get();
    }

    __AGENCY_ANNOTATION
    T get()
    {
      T result = std::move(*storage_);

      storage_.reset();

      return std::move(result);
    }

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return storage_;
    }

    __AGENCY_ANNOTATION
    void swap(asynchronous_state_impl& other)
    {
      storage_.swap(other.storage_);
    }

    __AGENCY_ANNOTATION
    storage_type& storage()
    {
      return storage_;
    }

  private:
    template<class, class, bool>
    friend class asynchronous_state_impl;

    storage_type storage_;
};


// when a type is empty, we can create instances on the fly upon dereference
template<class T>
struct empty_type_ptr : T
{
  using element_type = T;

  __AGENCY_ANNOTATION
  T& operator*()
  {
    return *this;
  }

  __AGENCY_ANNOTATION
  const T& operator*() const
  {
    return *this;
  }
};

template<>
struct empty_type_ptr<void> : unit_ptr {};


// zero storage optimization
template<class T, class Alloc>
class asynchronous_state_impl<T,Alloc,true>
{
  public:
    using value_type = T;
    using pointer = empty_type_ptr<T>;
    using storage_type = void;

    // constructs an invalid state
    __AGENCY_ANNOTATION
    asynchronous_state_impl() : valid_(false) {}

    // constructs an immediately ready state
    template<class OtherT,
             class = typename std::enable_if<
               std::is_constructible<T,OtherT&&>::value
             >::type>
    __AGENCY_ANNOTATION
    asynchronous_state_impl(construct_ready_t, OtherT&&) : valid_(true) {}

    __AGENCY_ANNOTATION
    asynchronous_state_impl(construct_ready_t) : valid_(true) {}

    // constructs a not ready state
    __AGENCY_ANNOTATION
    asynchronous_state_impl(construct_not_ready_t) : valid_(true) {}

    __AGENCY_ANNOTATION
    asynchronous_state_impl(asynchronous_state_impl&& other) : valid_(other.valid_)
    {
      other.valid_ = false;
    }

    // 1. allow moves to void states (this simply discards the state)
    // 2. allow moves to empty types if the type can be constructed from an empty argument list
    // 3. allow upcasts to a base T from a derived U
    template<class OtherT,
             class OtherAlloc,
             class T1 = T,
             class = typename std::enable_if<
               std::is_void<T1>::value ||
               (std::is_empty<T>::value && std::is_void<OtherT>::value && std::is_constructible<T>::value) ||
               std::is_base_of<T,OtherT>::value
             >::type>
    __AGENCY_ANNOTATION
    asynchronous_state_impl(asynchronous_state_impl<OtherT,OtherAlloc>&& other)
      : valid_(other.valid())
    {
      if(valid())
      {
        // invalidate the old state by calling .get() if it was valid when we received it
        other.get();
      }
    }

    __AGENCY_ANNOTATION
    asynchronous_state_impl& operator=(asynchronous_state_impl&& other)
    {
      valid_ = other.valid_;
      other.valid_ = false;

      return *this;
    }

    __AGENCY_ANNOTATION
    empty_type_ptr<T> data() const
    {
      return empty_type_ptr<T>();
    }

    __AGENCY_ANNOTATION
    T get()
    {
      valid_ = false;

      return get_impl(std::is_void<T>());
    }

    __AGENCY_ANNOTATION
    bool valid() const
    {
      return valid_;
    }

    __AGENCY_ANNOTATION
    void swap(asynchronous_state_impl& other)
    {
      bool other_valid_old = other.valid_;
      other.valid_ = valid_;
      valid_ = other_valid_old;
    }

  private:
    __AGENCY_ANNOTATION
    static T get_impl(std::false_type)
    {
      return T{};
    }

    __AGENCY_ANNOTATION
    static T get_impl(std::true_type)
    {
      return;
    }

    bool valid_;
};


// XXX this extra indirection is due to nvbug 1700337
//     eliminate this class when that bug is resolved
template<class T, class Alloc = std::allocator<T>>
class asynchronous_state : public asynchronous_state_impl<T,Alloc>
{
  public:
    using asynchronous_state_impl<T,Alloc>::asynchronous_state_impl;
};

  
} // end detail
} // end agency

