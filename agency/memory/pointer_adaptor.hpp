// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <iterator>
#include <memory>
#include <type_traits>


namespace agency
{
namespace detail
{


// has_binary_minus is used in the implementation of pointer_adaptor below
// XXX the reason it is defined outside of pointer_adaptor is because the GCC workaround
//     used below requires explicit specialization
template<class T1, class T2>
using binary_minus_t = decltype(std::declval<T1>() - std::declval<T2>());

// XXX workaround nvbug 2297457
//template<class T1,T2>
//using has_binary_minus = detail::is_detected<binary_minus_t, T1, T2>;

template<class T1, class T2>
struct has_binary_minus
{
  template<class U1, class U2,
           class = binary_minus_t<U1,U2>
          >
  static constexpr bool test(int) { return true; }

  template<class,class>
  static constexpr bool test(...) { return false; }

  static constexpr bool value = test<T1,T2>(0);
};

// XXX workaround GCC bug 87282
template<>
struct has_binary_minus<void*,void*>
{
  static constexpr bool value = false;
};


} // end detail


// this declaration of pointer_adaptor is for pointer_adaptor_reference's benefit
template<class T, class Accessor>
class pointer_adaptor;


template<class T, class Accessor>
class pointer_adaptor_reference : private Accessor
{
  private:
    // note that we derive from Accessor for the empty base class optimization
    using super_t = Accessor;
    using element_type = T;
    using value_type = typename std::remove_cv<element_type>::type;
    using accessor_type = Accessor;

    template<class U>
    using member_handle_type = typename U::handle_type;

    template<class U>
    using member_difference_type = typename U::difference_type;

    using handle_type = detail::detected_or_t<T*, member_handle_type, Accessor>;
  
  public:
    pointer_adaptor_reference() = default;
  
    pointer_adaptor_reference(const pointer_adaptor_reference&) = default;
  
    __AGENCY_ANNOTATION
    pointer_adaptor_reference(const handle_type& handle, const accessor_type& accessor)
      : super_t(accessor), handle_(handle)
    {}

  private:
    __AGENCY_ANNOTATION
    value_type load() const
    {
      return this->load(accessor(), handle_);
    }
  
  public:
    __AGENCY_ANNOTATION
    operator value_type () const
    {
      return load();
    }
  
    // address-of operator returns a pointer_adaptor
    __AGENCY_ANNOTATION
    pointer_adaptor<T,Accessor> operator&() const
    {
      return pointer_adaptor<T,Accessor>(handle_, accessor());
    }
  
    // the copy-assignment operator is const because it does not modify
    // the reference, even though it does modify the referent
    __AGENCY_ANNOTATION
    pointer_adaptor_reference operator=(const pointer_adaptor_reference& ref) const
    {
      copy_assign(ref);
      return *this;
    }
  
    template<__AGENCY_REQUIRES(
              std::is_assignable<element_type&, value_type>::value
            )>
    __AGENCY_ANNOTATION
    pointer_adaptor_reference operator=(const value_type& value) const
    {
      this->store(accessor(), handle_, value);
      return *this;
    }
  
    // this overload simply generates a diagnostic with the static_assert
    template<__AGENCY_REQUIRES(
              !std::is_assignable<element_type&, value_type>::value
            )>
    __AGENCY_ANNOTATION
    pointer_adaptor_reference operator=(const value_type&) const
    {
      static_assert(std::is_assignable<element_type&, value_type>::value, "pointer_adaptor element_type is not assignable.");
      return *this;
    }

    // equality
    // XXX this should only be enabled if value_type has operator==
    __AGENCY_ANNOTATION
    bool operator==(const value_type& value) const
    {
      return load() == value;
    }

    // inequality
    // XXX this should only be enabled if value_type has operator!=
    __AGENCY_ANNOTATION
    bool operator!=(const value_type& value) const
    {
      return load() != value;
    }

  private:
    template<class U, class Arg>
    using bracket_operator_t = decltype(std::declval<U>()[std::declval<Arg>()]);
  
    // XXX workaround nvbug 2297457
    //template<class U, class Arg>
    //using has_bracket_operator = detail::is_detected<bracket_operator_t, U, Arg>;

    template<class U, class Arg>
    struct has_bracket_operator
    {
      template<class V, class A,
               class = bracket_operator_t<V,A>
              >
      static constexpr bool test(int) { return true; }

      template<class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<U,Arg>(0);
    };

  public:
    // bracket operator
    template<class Arg,
             __AGENCY_REQUIRES(
               has_bracket_operator<T,Arg&&>::value
            )>
    __AGENCY_ANNOTATION
    bracket_operator_t<T,Arg&&> operator[](Arg&& arg) const
    {
      return load()[std::forward<Arg>(arg)];
    }
  
  private:
    __AGENCY_ANNOTATION
    const accessor_type& accessor() const
    {
      return *this;
    }
  
    __AGENCY_ANNOTATION
    accessor_type& accessor()
    {
      return *this;
    }
  
    template<__AGENCY_REQUIRES(
              std::is_assignable<element_type&, const element_type&>::value
            )>
    __AGENCY_ANNOTATION
    pointer_adaptor_reference copy_assign(const pointer_adaptor_reference& ref) const
    {
      this->store(accessor(), handle_, ref.handle_);
      return *this;
    }
  
    template<class A>
    using member_load_t = decltype(std::declval<A>().load(std::declval<handle_type>()));
  
    template<class A>
    using has_member_load = detail::is_detected_exact<value_type, member_load_t, A>; 
  
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(has_member_load<accessor_type>::value)>
    __AGENCY_ANNOTATION
    static value_type load(const accessor_type& accessor, const handle_type& handle)
    {
      return accessor.load(handle);
    }
  
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!has_member_load<accessor_type>::value and std::is_pointer<handle_type>::value)>
    __AGENCY_ANNOTATION
    static value_type load(const accessor_type& accessor, const handle_type& handle)
    {
      return *handle;
    }
  
  
    template<class A, class V>
    using member_store_t = decltype(std::declval<A>().store(std::declval<handle_type>(), std::declval<V>()));
  
    // XXX workaround nvbug 2297457
    //template<class V>
    //using accessor_has_member_store = detail::is_detected<member_store_t, accessor_type, V>;

    template<class U>
    struct accessor_has_member_store
    {
      template<class V,
               class = member_store_t<accessor_type, V>
              >
      static constexpr bool test(int) { return true; }

      template<class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<U>(0);
    };
  
  
    // store from an "immediate" element_type
    // this overload stores from an element_type using the accessor's .store() function, when available
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(accessor_has_member_store<const value_type&>::value)>
    __AGENCY_ANNOTATION
    static void store(const accessor_type& accessor, const handle_type& handle, const value_type& value)
    {
      accessor.store(handle, value);
    }
  
    // store from an "immediate" element_type
    // this overload stores from an element_type using simple raw pointer dereference
    // it is applicable only when the handle_type is a raw pointer
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!accessor_has_member_store<const value_type&>::value and std::is_pointer<handle_type>::value)>
    __AGENCY_ANNOTATION
    static void store(const accessor_type&, const handle_type& handle, const value_type& value)
    {
      *handle = value;
    }
  
    // "indirect" store from a handle_type
    // this overload stores from a handle_type using the accessor's .store() function, when available
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(accessor_has_member_store<handle_type>::value)>
    __AGENCY_ANNOTATION
    static void store(const accessor_type& accessor, const handle_type& dst, const handle_type& src)
    {
      accessor.store(dst, src);
    }
  
    // "indirect" store from a handle_type
    // this overload stores from a handle_type using a temporary intermediate value
    // it first uses load() and then the element_type version of store()
    template<__AGENCY_REQUIRES(!accessor_has_member_store<handle_type>::value and !std::is_pointer<handle_type>::value)>
    __AGENCY_ANNOTATION
    static void store(const accessor_type& accessor, const handle_type& dst, const handle_type& src)
    {
      // first load the source into a temporary
      element_type value = load(accessor, src);
  
      // now store the temporary value
      store(accessor, dst, value);
    }
  
    // "indirect" store from a handle_type
    // this overload stores from a handle_type using simple raw pointer dereference
    // it is applicable only when the handle_type is a raw pointer
    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!accessor_has_member_store<handle_type>::value and std::is_pointer<handle_type>::value)>
    __AGENCY_ANNOTATION
    static void store(const accessor_type& accessor, const handle_type& dst, const handle_type& src)
    {
      *dst = *src;
    }
  
    handle_type handle_;
}; // end pointer_adaptor_reference


template<class T, class Accessor>
class pointer_adaptor : private Accessor
{
  private:
    // note that we derive from Accessor for the empty base class optimization
    using super_t = Accessor;

    template<class U>
    using member_handle_type = typename U::handle_type;

    template<class U>
    using member_difference_type = typename U::difference_type;

  public:
    // pointer_traits member types
    using element_type = T;
    using accessor_type = Accessor;
    using handle_type = detail::detected_or_t<T*, member_handle_type, Accessor>;

    // XXX if member difference_type doesn't exist, we should use the result of difference_to(), if it exists, otherwise we should use std::ptrdiff_t
    using difference_type = detail::detected_or_t<std::ptrdiff_t, member_difference_type, Accessor>;

    // additional iterator_traits member types
    using value_type = typename std::remove_cv<element_type>::type;
    using iterator_category = std::random_access_iterator_tag;
    using pointer = pointer_adaptor;
    using reference = pointer_adaptor_reference<T,Accessor>;

    pointer_adaptor() = default;

  private:
    template<class U>
    using member_null_handle_t = decltype(std::declval<U>().null_handle());

    template<class U>
    using has_member_null_handle = detail::is_detected_exact<handle_type, member_null_handle_t, accessor_type>;

    template<__AGENCY_REQUIRES(has_member_null_handle<accessor_type>::value)>
    __AGENCY_ANNOTATION
    static handle_type null_handle(const accessor_type& accessor)
    {
      return accessor.null_handle();
    }

    template<__AGENCY_REQUIRES(!has_member_null_handle<accessor_type>::value && std::is_constructible<handle_type, std::nullptr_t>::value)>
    __AGENCY_ANNOTATION
    static handle_type null_handle(const accessor_type&)
    {
      return nullptr;
    }

  public:
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    pointer_adaptor(std::nullptr_t) noexcept
      : pointer_adaptor(null_handle(accessor_type()))
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    pointer_adaptor(const pointer_adaptor& other)
      : pointer_adaptor(other.get(), other.accessor())
    {}

    pointer_adaptor& operator=(const pointer_adaptor&) = default;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    explicit pointer_adaptor(const handle_type& h) noexcept
      : pointer_adaptor(h, accessor_type())
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    pointer_adaptor(const handle_type& h, const accessor_type& a) noexcept
      : super_t(a), handle_(h)
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    pointer_adaptor(const handle_type& h, accessor_type&& a) noexcept
      : super_t(std::move(a)), handle_(h)
    {}

    template<class U, class OtherAccessor,
             __AGENCY_REQUIRES(
               std::is_convertible<U*, T*>::value // This first requirement disables unreasonable conversions, e.g. pointer_adaptor<void,...> -> pointer_adaptor<int,...>
               and std::is_convertible<typename pointer_adaptor<U,OtherAccessor>::handle_type, handle_type>::value
               and std::is_convertible<OtherAccessor, accessor_type>::value
            )>
    __AGENCY_ANNOTATION
    pointer_adaptor(const pointer_adaptor<U,OtherAccessor>& other)
      : pointer_adaptor(other.get(), other.accessor())
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    ~pointer_adaptor() {}

    // returns the underlying handle
    __AGENCY_ANNOTATION
    const handle_type& get() const noexcept
    {
      return handle_;
    }

    // returns the accessor
    __AGENCY_ANNOTATION
    Accessor& accessor() noexcept
    {
      return *this;
    }

    // returns the accessor
    __AGENCY_ANNOTATION
    const Accessor& accessor() const noexcept
    {
      return *this;
    }

    // conversion to bool
    __AGENCY_ANNOTATION
    explicit operator bool() const noexcept
    {
      return *this != nullptr;
    }

    // dereference
    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return reference(get(), accessor());
    }

    // subscript
    __AGENCY_ANNOTATION
    reference operator[](difference_type i) const
    {
      return *(*this + i);
    }

    // pre-increment
    __AGENCY_ANNOTATION
    pointer_adaptor& operator++()
    {
      this->advance(accessor(), handle_, 1);
      return *this;
    }

    // pre-decrement
    __AGENCY_ANNOTATION
    pointer_adaptor& operator--()
    {
      this->advance(accessor(), handle_, -1);
      return *this;
    }

    // post-increment
    __AGENCY_ANNOTATION
    pointer_adaptor operator++(int)
    {
      pointer_adaptor result = *this;
      operator++();
      return result;
    }

    // post-decrement
    __AGENCY_ANNOTATION
    pointer_adaptor operator--(int)
    {
      pointer_adaptor result = *this;
      operator--();
      return result;
    }

    // plus
    __AGENCY_ANNOTATION
    pointer_adaptor operator+(difference_type n) const
    {
      pointer_adaptor result = *this;
      result += n;
      return result;
    }

    // minus
    __AGENCY_ANNOTATION
    pointer_adaptor operator-(difference_type n) const
    {
      pointer_adaptor result = *this;
      result -= n;
      return result;
    }

    // plus-equal
    __AGENCY_ANNOTATION
    pointer_adaptor& operator+=(difference_type n)
    {
      this->advance(accessor(), handle_, n);
      return *this;
    }

    // minus-equal
    __AGENCY_ANNOTATION
    pointer_adaptor& operator-=(difference_type n)
    {
      this->advance(accessor(), handle_, -n);
      return *this;
    }

    // difference
    __AGENCY_ANNOTATION
    difference_type operator-(const pointer_adaptor& other) const noexcept
    {
      return this->distance_to(accessor(), other.get(), get());
    }

    // equality
    __AGENCY_ANNOTATION
    bool operator==(const pointer_adaptor& other) const noexcept
    {
      return this->equal(accessor(), handle_, other.handle_);
    }

    __AGENCY_ANNOTATION
    friend bool operator==(const pointer_adaptor& self, std::nullptr_t) noexcept
    {
      return self.equal(self.accessor(), self.handle_, null_handle(self.accessor()));
    }

    __AGENCY_ANNOTATION
    friend bool operator==(std::nullptr_t, const pointer_adaptor& self) noexcept
    {
      return self.equal(self.accessor(), null_handle(self.accessor(), self.handle_));
    }

    // inequality
    __AGENCY_ANNOTATION
    bool operator!=(const pointer_adaptor& other) const noexcept
    {
      return !operator==(other);
    }

    __AGENCY_ANNOTATION
    friend bool operator!=(const pointer_adaptor& self, std::nullptr_t) noexcept
    {
      return !(self == nullptr);
    }

    __AGENCY_ANNOTATION
    friend bool operator!=(std::nullptr_t, const pointer_adaptor& self) noexcept
    {
      return !(nullptr == self);
    }

    // less
    __AGENCY_ANNOTATION
    bool operator<(const pointer_adaptor& other) const noexcept
    {
      return this->less(accessor(), handle_, other.handle_);
    }

    // lequal
    __AGENCY_ANNOTATION
    bool operator<=(const pointer_adaptor& other) const noexcept
    {
      return this->lequal(accessor(), handle_, other.handle_);
    }

  private:
    template<class U>
    using member_advance_t = decltype(std::declval<U>().advance(std::declval<handle_type&>(), std::declval<difference_type>()));

    // XXX workaround nvbug 2297457
    //template<class U>
    //using has_member_advance = detail::is_detected<member_advance_t, accessor_type>; 

    template<class U>
    struct has_member_advance
    {
      template<class V,
               class = member_advance_t<V>
              >
      static constexpr bool test(int) { return true; }

      template<class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<U>(0);
    };

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(has_member_advance<accessor_type>::value)>
    __AGENCY_ANNOTATION
    static void advance(accessor_type& accessor, handle_type& handle, difference_type n)
    {
      accessor.advance(handle, n);
    }

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!has_member_advance<accessor_type>::value && std::is_pointer<handle_type>::value)>
    __AGENCY_ANNOTATION
    static void advance(accessor_type&, handle_type& handle, difference_type n)
    {
      handle += n;
    }


    template<class U>
    using member_distance_to_t = decltype(std::declval<U>().distance_to(std::declval<handle_type>(), std::declval<handle_type>()));

    // XXX workaround nvbug 2297457
    //template<class U>
    //using has_member_distance_to = detail::is_detected<member_distance_to, accessor_type>;

    template<class U>
    struct has_member_distance_to
    {
      template<class V,
               class = member_distance_to_t<V>
              >
      static constexpr bool test(int) { return true; }

      template<class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<U>(0);
    };

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(detail::has_binary_minus<handle_type,handle_type>::value)>
    __AGENCY_ANNOTATION
    static difference_type distance_to(const accessor_type& accessor, const handle_type& from, const handle_type& to)
    {
      return to - from;
    }

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!detail::has_binary_minus<handle_type,handle_type>::value and has_member_distance_to<accessor_type>::value)>
    __AGENCY_ANNOTATION
    static difference_type distance_to(const accessor_type& accessor, const handle_type& from, const handle_type& to)
    {
      return accessor.distance_to(from, to);
    }


    template<class T1, class T2>
    using less_t = decltype(std::declval<T1>() < std::declval<T2>());

    // XXX workaround nvbug 2297457
    //template<class T1,class T2>
    //using has_less = detail::is_detected<less_t, T1, T2>;

    template<class T1, class T2>
    struct has_less
    {
      template<class U1, class U2,
               class = less_t<U1,U2>
              >
      static constexpr bool test(int) { return true; }

      template<class,class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<T1,T2>(0);
    };


    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(has_less<handle_type,handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool less(const accessor_type&, const handle_type& lhs, const handle_type& rhs)
    {
      return lhs < rhs;
    }

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!has_less<handle_type,handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool less(const accessor_type& accessor, const handle_type& lhs, const handle_type& rhs)
    {
      return 0 < distance_to(accessor, lhs, rhs);
    }


    template<class T1, class T2>
    using lequal_t = decltype(std::declval<T1>() <= std::declval<T2>());

    // XXX workaround nvbug 2297457
    //template<class T1, class T2>
    //using has_lequal = detail::is_detected<lequal_t, T1, T2>;

    template<class T1, class T2>
    struct has_lequal
    {
      template<class U1, class U2,
               class = lequal_t<U1,U2>
              >
      static constexpr bool test(int) { return true; }

      template<class,class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<T1,T2>(0);
    };


    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(has_lequal<handle_type, handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool lequal(const accessor_type&, const handle_type& lhs, const handle_type& rhs)
    {
      return lhs <= rhs;
    }

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!has_lequal<handle_type, handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool lequal(const accessor_type& accessor, const handle_type& lhs, const handle_type& rhs)
    {
      return 0 <= distance_to(accessor, lhs, rhs);
    }


    template<class T1, class T2>
    using equal_t = decltype(std::declval<T1>() == std::declval<T2>());

    // XXX workaround nvbug 2297457
    //template<class T1, class T2>
    //using has_equal = detail::is_detected<equal_t, T1, T2>;

    template<class T1, class T2>
    struct has_equal
    {
      template<class U1, class U2,
               class = equal_t<U1,U2>
              >
      static constexpr bool test(int) { return true; }

      template<class,class>
      static constexpr bool test(...) { return false; }

      static constexpr bool value = test<T1,T2>(0);
    };


    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(has_equal<handle_type, handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool equal(const accessor_type&, const handle_type& lhs, const handle_type& rhs)
    {
      return lhs == rhs;
    }

    __agency_exec_check_disable__
    template<__AGENCY_REQUIRES(!has_equal<handle_type, handle_type>::value)>
    __AGENCY_ANNOTATION
    static bool equal(const accessor_type& accessor, const handle_type& lhs, const handle_type& rhs)
    {
      return distance_to(accessor, lhs, rhs) == 0;
    }

    handle_type handle_;
};


} // end agency


// specialize std::pointer_traits for pointer_adaptor
namespace std
{


template<class T, class Accessor>
struct pointer_traits<agency::pointer_adaptor<T,Accessor>>
{
  using pointer = typename agency::pointer_adaptor<T,Accessor>::pointer;
  using element_type = typename agency::pointer_adaptor<T,Accessor>::element_type;
  using difference_type = typename agency::pointer_adaptor<T,Accessor>::difference_type;

  template<__AGENCY_REQUIRES(
             std::is_pointer<
               typename agency::pointer_adaptor<T,Accessor>::handle_type
             >::value
          )>
  __AGENCY_ANNOTATION
  static typename agency::pointer_adaptor<T,Accessor>::handle_type
    to_address(const pointer ptr) noexcept
  {
    return ptr.get();
  }

  template<__AGENCY_REQUIRES(
             std::is_pointer<
               typename agency::pointer_adaptor<T,Accessor>::handle_type
             >::value
          )>
  __AGENCY_ANNOTATION
  static agency::pointer_adaptor<T,Accessor> pointer_to(element_type& r) noexcept
  {
    return {&r};
  }

  template<class U>
  using rebind = agency::pointer_adaptor<U,Accessor>;
};
  

} // end std

