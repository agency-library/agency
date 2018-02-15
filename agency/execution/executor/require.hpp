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
#include <agency/detail/type_traits.hpp>
#include <agency/detail/static_const.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_has_static_property.hpp>
#include <agency/execution/executor/executor_traits/detail/has_require_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_require_free_function.hpp>
#include <utility>


namespace agency
{
namespace detail
{


// this is the type of agency::require
struct require_t
{
  // these overloads of operator() are listed in decreasing priority order

  template<class E, class P,
           __AGENCY_REQUIRES(
             executor_has_static_property<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&&) const ->
    decltype(std::forward<E&&>(e))
  {
    return std::forward<E&&>(e);
  }


  // member function e.require(p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(
             !executor_has_static_property<decay_t<E>,decay_t<P>>::value and
             has_require_member<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(std::forward<E>(e).require(std::forward<P>(p)))
  {
    return std::forward<E>(e).require(std::forward<P>(p));
  }


  // free function require(e,p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(
             !executor_has_static_property<decay_t<E>,decay_t<P>>::value and
             !has_require_member<decay_t<E>,decay_t<P>>::value and
             has_require_free_function<decay_t<E>,decay_t<P>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(require(std::forward<E>(e),std::forward<P>(p)))
  {
    return require(std::forward<E>(e), std::forward<P>(p));
  }


  // variadic overload
  template<class Executor, class Property, class... Properties>
  __AGENCY_ANNOTATION
  constexpr auto operator()(Executor&& ex, Property&& prop, Properties&&... props) const ->
    decltype(operator()(operator()(std::forward<Executor>(ex), std::forward<Property>(prop)), std::forward<Properties>(props)...))
  {
    // recurse
    return operator()(operator()(std::forward<Executor>(ex), std::forward<Property>(prop)), std::forward<Properties>(props)...);
  }
};


} // end detail


namespace
{


#ifndef __CUDA_ARCH__
constexpr auto const& require = detail::static_const<detail::require_t>::value;
#else
// __device__ functions cannot access global variables so make require a __device__ variable in __device__ code
const __device__ detail::require_t require;
#endif


} // end anonymous namespace
} // end agency

