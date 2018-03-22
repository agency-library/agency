/// \file
/// \brief Contains definition of replace_executor.
///

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
#include <agency/detail/static_const.hpp>
#include <agency/execution/executor/executor_traits/detail/has_associated_executor_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_associated_executor_free_function.hpp>


namespace agency
{
namespace detail
{


// this is the type of agency::associated_executor
struct associated_executor_t
{
  // member function x.associated_executor() overload
  template<class T,
           __AGENCY_REQUIRES(
             has_associated_executor_member<decay_t<T>>::value
           )>
  __AGENCY_ANNOTATION
  constexpr auto operator()(T&& x) const ->
    decltype(std::forward<T>(x).associated_executor())
  {
    return std::forward<T>(x).associated_executor();
  }

  // free function associated_executor(val) overload
  template<class T,
           __AGENCY_REQUIRES(
             !has_associated_executor_member<decay_t<T>>::value and
             has_associated_executor_free_function<decay_t<T>>::value
           )>
  constexpr auto operator()(T&& x) const ->
    decltype(associated_executor(std::forward<T>(x)))
  {
    return associated_executor(std::forward<T>(x));
  }
};

} // end detail


namespace
{


// define the associated_executor customization point object
#ifndef __CUDA_ARCH__
constexpr auto const& associated_executor = detail::static_const<detail::associated_executor_t>::value;
#else
// CUDA __device__ functions cannot access global variables so make associated_executor a __device__ variable in __device__ code
const __device__ detail::associated_executor_t associated_executor;
#endif

} // end anonymous namespace


} // end agency

