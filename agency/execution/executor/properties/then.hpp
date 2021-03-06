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
#include <agency/execution/executor/detail/adaptors/basic_executor_adaptor.hpp>
#include <agency/execution/executor/executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/is_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_executor.hpp>
#include <agency/execution/executor/detail/execution_functions/bulk_then_execute.hpp>
#include <agency/execution/executor/detail/execution_functions/then_execute.hpp>
#include <utility>


namespace agency
{
namespace detail
{


template<class Executor>
class then_executor : public basic_executor_adaptor<Executor>
{
  private:
    using super_t = basic_executor_adaptor<Executor>;

  public:
    then_executor() = default;

    __AGENCY_ANNOTATION
    then_executor(const Executor& ex) noexcept : super_t{ex} {}

    template<class Function, class Future,
             __AGENCY_REQUIRES(
               is_single_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
      then_execute(Function&& f, Future& fut) const
    {
      return detail::then_execute(super_t::base_executor(), std::forward<Function>(f), fut);
    }

    template<class Function, class Future, class ResultFactory, class... SharedFactories,
             __AGENCY_REQUIRES(is_bulk_executor<Executor>::value),
             __AGENCY_REQUIRES(executor_execution_depth<Executor>::value == sizeof...(SharedFactories))
            >
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, executor_shape_t<Executor> shape, Future& predecessor, ResultFactory result_factory, SharedFactories... shared_factories)
    {
      return detail::bulk_then_execute(super_t::base_executor(), f, shape, predecessor, result_factory, shared_factories...);
    }
}; // end then_executor


} // end detail


struct then_t
{
  constexpr static bool is_requirable = true;
  constexpr static bool is_preferable = false;

  // Agency is a C++11-compatible library,
  // so we can't implement static_query_v as a variable template
  // use a constexpr static function instead
  template<class E>
  __AGENCY_ANNOTATION
  constexpr static bool static_query()
  {
    return detail::is_then_executor<E>::value;
  }
  
  template<class Executor>
  __AGENCY_ANNOTATION
  friend detail::then_executor<Executor> require(Executor ex, then_t)
  {
    return detail::then_executor<Executor>{ex};
  }
};


// define the property object

#ifndef __CUDA_ARCH__
constexpr then_t then{};
#else
// CUDA __device__ functions cannot access global variables so make then a __device__ variable in __device__ code
const __device__ then_t then;
#endif


} // end agency

