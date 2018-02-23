// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <stddef.h> // XXX instead of <cstddef> to WAR clang issue
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class T>
struct tuple_use_empty_base_class_optimization
  : std::integral_constant<
      bool,
      std::is_empty<T>::value
#if __cplusplus >= 201402L
      && !std::is_final<T>::value
#endif
    >
{};


template<class T, bool = tuple_use_empty_base_class_optimization<T>::value>
class tuple_leaf_base
{
  public:
    __agency_exec_check_disable__
    tuple_leaf_base() = default;

    __agency_exec_check_disable__
    template<class U>
    __AGENCY_ANNOTATION
    tuple_leaf_base(U&& arg) : val_(std::forward<U>(arg)) {}

    __agency_exec_check_disable__
    ~tuple_leaf_base() = default;

    __AGENCY_ANNOTATION
    const T& const_get() const
    {
      return val_;
    }

    __AGENCY_ANNOTATION
    T& mutable_get()
    {
      return val_;
    }

  private:
    T val_;
};


template<class T>
class tuple_leaf_base<T,true> : public T
{
  public:
    tuple_leaf_base() = default;

    template<class U>
    __AGENCY_ANNOTATION
    tuple_leaf_base(U&& arg) : T(std::forward<U>(arg)) {}

    __AGENCY_ANNOTATION
    const T& const_get() const
    {
      return *this;
    }
  
    __AGENCY_ANNOTATION
    T& mutable_get()
    {
      return *this;
    }
};


template<std::size_t I, class T>
class tuple_leaf : public tuple_leaf_base<T>
{
  private:
    using super_t = tuple_leaf_base<T>;


  public:
    tuple_leaf() = default;


    template<class U,
             __AGENCY_REQUIRES(
               std::is_constructible<T,U>::value
             )>
    __AGENCY_ANNOTATION
    tuple_leaf(U&& arg) : super_t(std::forward<U>(arg)) {}


    __AGENCY_ANNOTATION
    tuple_leaf(const tuple_leaf& other) : super_t(other.const_get()) {}


    __AGENCY_ANNOTATION
    tuple_leaf(tuple_leaf&& other) : super_t(std::forward<T>(other.mutable_get())) {}


    template<class U,
             __AGENCY_REQUIRES(
               std::is_constructible<T,const U&>::value
             )>
    __AGENCY_ANNOTATION
    tuple_leaf(const tuple_leaf<I,U>& other) : super_t(other.const_get()) {}


    // converting move-constructor
    // note the use of std::forward<U> here to allow construction of T from U&&
    template<class U,
             __AGENCY_REQUIRES(
               std::is_constructible<T,U&&>::value
             )>
    __AGENCY_ANNOTATION
    tuple_leaf(tuple_leaf<I,U>&& other) : super_t(std::forward<U>(other.mutable_get())) {}


    __agency_exec_check_disable__
    template<class U,
             __AGENCY_REQUIRES(
               std::is_assignable<T,U>::value
             )>
    __AGENCY_ANNOTATION
    tuple_leaf& operator=(const tuple_leaf<I,U>& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }

    
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    tuple_leaf& operator=(const tuple_leaf& other)
    {
      this->mutable_get() = other.const_get();
      return *this;
    }


    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    tuple_leaf& operator=(tuple_leaf&& other)
    {
      this->mutable_get() = std::forward<T>(other.mutable_get());
      return *this;
    }


    __agency_exec_check_disable__
    template<class U,
             __AGENCY_REQUIRES(
               std::is_assignable<T,U&&>::value
             )>
    __AGENCY_ANNOTATION
    tuple_leaf& operator=(tuple_leaf<I,U>&& other)
    {
      this->mutable_get() = std::forward<U>(other.mutable_get());
      return *this;
    }


    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    int swap(tuple_leaf& other)
    {
      using std::swap;
      swap(this->mutable_get(), other.mutable_get());
      return 0;
    }
};


} // end detail
} // end agency

