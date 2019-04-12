#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unit.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{


// XXX this should probably go in unit.hpp
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


// when a type is empty, we can create instances on the fly upon dereference
template<class T>
struct empty_type_ptr
{
  using element_type = T;
  
  empty_type_ptr() = default;
  
  empty_type_ptr(const empty_type_ptr&) = default;

  empty_type_ptr(std::nullptr_t) {}
  
  template<class... Args,
           __AGENCY_REQUIRES(
             std::is_constructible<T,Args&&...>::value
           )>
  __AGENCY_ANNOTATION
  empty_type_ptr(Args&&... args)
  {
    // this evaluates T's copy constructor's effects, but nothing is stored
    // because both T and empty_type_ptr are empty types
    new (this) T(std::forward<Args>(args)...);
  }
  
  __AGENCY_ANNOTATION
  T& operator*()
  {
    return *reinterpret_cast<T*>(this);
  }
  
  __AGENCY_ANNOTATION
  const T& operator*() const
  {
    return *reinterpret_cast<const T*>(this);
  }
  
  // even though T is empty and there is nothing to swap,
  // swap(T,T) may have effects, so call it
  __AGENCY_ANNOTATION
  void swap(empty_type_ptr& other)
  {
    detail::adl_swap(**this, *other);
  }
};


template<>
struct empty_type_ptr<void> : unit_ptr
{
  empty_type_ptr() = default;

  empty_type_ptr(const empty_type_ptr&) = default;

  // allow copy construction from empty_type_ptr<T>
  // this is analogous to casting a T to void
  template<class T>
  __AGENCY_ANNOTATION
  empty_type_ptr(const empty_type_ptr<T>&)
    : empty_type_ptr()
  {}

  __AGENCY_ANNOTATION
  void swap(empty_type_ptr&) const
  {
    // swapping a void has no effect
  }
};


template<class T>
struct empty_type_allocator
{
  using value_type = T;
  using pointer = empty_type_ptr<T>;

  empty_type_allocator() = default;
  empty_type_allocator(const empty_type_allocator&) = default;

  template<class U>
  __AGENCY_ANNOTATION
  empty_type_allocator(const empty_type_allocator<U>&) {}

  __AGENCY_ANNOTATION
  static pointer allocate(size_t)
  {
    return pointer{};
  }

  __AGENCY_ANNOTATION
  static pointer deallocate(pointer, size_t) {}

  template<class U, class... Args>
  __AGENCY_ANNOTATION
  static void construct(empty_type_ptr<U>, Args&&... args)
  {
    // empty_type_ptr's ctor calls U's constructor
    empty_type_ptr<U>(std::forward<Args>(args)...);
  }

  template<class U>
  __AGENCY_ANNOTATION
  static void destroy(empty_type_ptr<U> ptr)
  {
    // call U's destructor
    ptr->~U();
  }

  __AGENCY_ANNOTATION
  bool operator==(const empty_type_allocator&) const
  {
    return true;
  }

  __AGENCY_ANNOTATION
  bool operator!=(const empty_type_allocator&) const
  {
    return false;
  }
};


} // end detail
} // end agency

