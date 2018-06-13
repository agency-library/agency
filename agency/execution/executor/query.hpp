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
#include <agency/execution/executor/executor_traits/detail/has_static_query.hpp>
#include <agency/execution/executor/executor_traits/detail/has_query_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_query_free_function.hpp>
#include <agency/detail/type_traits.hpp>


namespace agency
{
namespace detail
{


// this is the type of agency::query
struct query_t
{
  // these overloads of operator() are listed in decreasing priority order

  // P::static_query<E>() overload
  __agency_exec_check_disable__
  template<class E, class P,
           __AGENCY_REQUIRES(has_static_query<decay_t<P>,decay_t<E>>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(decay_t<P>::template static_query<decay_t<E>>())
  {
    return decay_t<P>::template static_query<decay_t<E>>();
  }

  
  // member function e.query(p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(!has_static_query<decay_t<P>, decay_t<E>>::value),
           __AGENCY_REQUIRES(has_query_member<decay_t<E>, decay_t<P>>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(std::forward<E>(e).query(std::forward<P>(p)))
  {
    return std::forward<E>(e).query(std::forward<P>(p));
  }


  // free function query(e,p) overload
  template<class E, class P,
           __AGENCY_REQUIRES(!has_static_query<decay_t<P>, decay_t<E>>::value),
           __AGENCY_REQUIRES(!has_query_member<decay_t<E>, decay_t<P>>::value),
           __AGENCY_REQUIRES(has_query_free_function<decay_t<E>, decay_t<P>>::value)
          >
  __AGENCY_ANNOTATION
  constexpr auto operator()(E&& e, P&& p) const ->
    decltype(query(std::forward<E>(e),std::forward<P>(p)))
  {
    return query(std::forward<E>(e), std::forward<P>(p));
  }
};


} // end detail


namespace
{


// define the query customization point object

#ifndef __CUDA_ARCH__
constexpr auto const& query = detail::static_const<detail::query_t>::value;
#else
// CUDA __device__ functions cannot access global variables so make query a __device__ variable in __device__ code
const __device__ detail::query_t query;
#endif


} // end anonymous namespace


} // end agency

