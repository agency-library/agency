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
#include <agency/detail/type_traits.hpp>
#include <agency/future/future_traits/is_future.hpp>
#include <agency/future/future_traits/future_result.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{


template<class T>
struct dummy_functor
{
  void operator()(T&) {}
};

template<>
struct dummy_functor<void>
{
  void operator()() {}
};


template<class T, class Future>
using then_execute_member_t = decltype(std::declval<const T&>().then_execute(dummy_functor<future_result_t<Future>>(), std::declval<Future&>()));


template<class T, class Future>
struct has_then_execute_member
{
  template<class U,
           class ThenExecuteResult = detected_t<then_execute_member_t, U, Future>
           , class = typename std::enable_if<
             is_future<ThenExecuteResult>::value
           >::type
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  static const bool value = decltype(test<T>(0))::value;
};


} // end detail
} // end agency

