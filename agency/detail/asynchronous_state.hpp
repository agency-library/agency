#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unit.hpp>
#include <agency/memory/detail/unique_ptr.hpp>
#include <agency/memory/allocator/detail/allocator_traits/is_allocator.hpp>
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


// XXX asynchronous_state should be templated on Allocator instead of Deleter

// XXX should try to collapse the implementation of asynchronous_state as much as possible between the two specializations
// XXX the default value of Deleter should be some polymorphic deleter type
// XXX the default state of the polymorphic deleter type should be an instance of default_delete<T>
template<class T,
         class Allocator = std::allocator<T>,
         bool requires_storage = state_requires_storage<T>::value>
class asynchronous_state
{
  public:
    static_assert(is_allocator<Allocator>::value, "Allocator is not an allocator.");

    using value_type = T;
    using storage_type = unique_ptr<T,allocation_deleter<Allocator>>;
    using pointer = typename storage_type::pointer;

    // constructs an invalid state
    __AGENCY_ANNOTATION
    asynchronous_state() = default;

    // constructs an immediately ready state
    __agency_exec_check_disable__
    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args...>::value
             >::type
            >
    __AGENCY_ANNOTATION
    asynchronous_state(construct_ready_t, const Allocator& allocator, Args&&... ready_args)
      : storage_(allocate_unique<T>(allocator, std::forward<Args>(ready_args)...))
    {}

    // constructs a not ready state from a pointer to the result and an allocator which can delete & deallocate the pointer
    template<class OtherAllocator,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 Allocator, OtherAllocator
               >::value
             )>
    __AGENCY_ANNOTATION
    asynchronous_state(construct_not_ready_t, pointer ptr, const OtherAllocator& allocator)
      : storage_(ptr, allocation_deleter<Allocator>(allocator))
    {}

    // constructs a not ready state
    // XXX we should avoid creating an object here
    //     instead, we should just create it uninitialized
    // XXX the destructor should check whether the state requires destruction
    __AGENCY_ANNOTATION
    asynchronous_state(construct_not_ready_t, const Allocator& allocator)
      : asynchronous_state(construct_ready, allocator, T{})
    {}

    __AGENCY_ANNOTATION
    asynchronous_state(asynchronous_state&& other) = default;

    template<class OtherT,
             class OtherAllocator,
             class = typename std::enable_if<
               std::is_constructible<storage_type, typename asynchronous_state<OtherT,OtherAllocator>::storage_type&&>::value
             >::type
            >
    __AGENCY_ANNOTATION
    asynchronous_state(asynchronous_state<OtherT,OtherAllocator>&& other)
      : storage_(std::move(other.storage_))
    {}

    __AGENCY_ANNOTATION
    asynchronous_state& operator=(asynchronous_state&&) = default;

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
    void swap(asynchronous_state& other)
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
    friend class asynchronous_state;

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
template<class T, class Allocator>
class asynchronous_state<T,Allocator,true>
{
  public:
    static_assert(is_allocator<Allocator>::value, "Allocator is not an allocator.");

    using value_type = T;
    using pointer = empty_type_ptr<T>;
    using storage_type = void;

    // constructs an invalid state
    __AGENCY_ANNOTATION
    asynchronous_state() : valid_(false) {}

    // constructs an immediately ready state
    // the allocator is ignored because this state requires no storage
    template<class OtherAllocator,
             class OtherT,
             __AGENCY_REQUIRES(
               std::is_constructible<Allocator,OtherAllocator>::value
             ),
             __AGENCY_REQUIRES(
               std::is_constructible<T,OtherT&&>::value
             )>
    __AGENCY_ANNOTATION
    asynchronous_state(construct_ready_t, const OtherAllocator&, OtherT&&) : valid_(true) {}

    // constructs an immediately ready state
    // the allocator is ignored because this state requires no storage
    template<class OtherAllocator,
             __AGENCY_REQUIRES(std::is_constructible<Allocator,OtherAllocator>::value)>
    __AGENCY_ANNOTATION
    asynchronous_state(construct_ready_t, const OtherAllocator&) : valid_(true) {}

    // constructs a not ready state
    // the allocator is ignored because this state requires no storage
    template<class OtherAllocator,
             __AGENCY_REQUIRES(std::is_constructible<Allocator,OtherAllocator>::value)>
    __AGENCY_ANNOTATION
    asynchronous_state(construct_not_ready_t, const OtherAllocator&) : valid_(true) {}

    __AGENCY_ANNOTATION
    asynchronous_state(asynchronous_state&& other) : valid_(other.valid_)
    {
      other.valid_ = false;
    }

    // 1. allow moves to void states (this simply discards the state)
    // 2. allow moves to empty types if the type can be constructed from an empty argument list
    // 3. allow upcasts to a base T from a derived U
    template<class OtherT,
             class OtherAllocator,
             class T1 = T,
             class = typename std::enable_if<
               std::is_void<T1>::value ||
               (std::is_empty<T>::value && std::is_void<OtherT>::value && std::is_constructible<T>::value) ||
               std::is_base_of<T,OtherT>::value
             >::type>
    __AGENCY_ANNOTATION
    asynchronous_state(asynchronous_state<OtherT,OtherAllocator>&& other)
      : valid_(other.valid())
    {
      if(valid())
      {
        // invalidate the old state by calling .get() if it was valid when we received it
        other.get();
      }
    }

    __AGENCY_ANNOTATION
    asynchronous_state& operator=(asynchronous_state&& other)
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
    void swap(asynchronous_state& other)
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

  
} // end detail
} // end agency

