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
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/is_asynchronous_executor.hpp>
#include <agency/future.hpp>
#include <utility>


namespace agency
{
namespace detail
{


// this class wraps a base executor and forwards calls
// to execution functions to the base, if they exist
// XXX also need to forward .require() and .prefer()
template<class Executor>
class basic_executor_adaptor
{
  private:
    // XXX nomerge
    // eliminate mutable once the execution functions of Agency's executors are const
    mutable Executor base_executor_;

  protected:
    // XXX nomerge
    // XXX this function should return const Executor& 
    // XXX currently, Agency's executors are not shallow-const as intended by P0443's design
    // XXX this function should return the correct type of reference once P0443's material has been fully incorporated and
    //     the execution functions of Agency's executors are const
    __AGENCY_ANNOTATION
    Executor& base_executor() const
    {
      return const_cast<Executor&>(base_executor_);
    }

  public:
    template<class T>
    using future = executor_future_t<Executor,T>;

    // XXX need to publicize the other executor member typedefs such as execution_category, etc.

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_executor_adaptor(const Executor& base) noexcept : base_executor_{base} {}

    __agency_exec_check_disable__
    template<class Function,
             __AGENCY_REQUIRES(is_twoway_executor<Executor>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute(Function&& f) const
    {
      return base_executor_.twoway_execute(std::forward<Function>(f));
    }

    // XXX nomerge
    // XXX eliminate this once Agency's executors have been ported to P0443's interface
    //     i.e., functions named .async_execute() are renamed .twoway_execute()
    __agency_exec_check_disable__
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<Executor>::value and
               is_asynchronous_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute(Function&& f) const
    {
      return base_executor_.async_execute(std::forward<Function>(f));
    }

    __agency_exec_check_disable__
    template<class Function, class Future,
             __AGENCY_REQUIRES(is_then_executor<Executor>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_continuation_t<decay_t<Function>, Future>>
      then_execute(Function&& f, Future& fut) const
    {
      return base_executor_.then_execute(std::forward<Function>(f), fut);
    }

    //__agency_exec_check_disable__
    //template<class Function, class Shape, class ResultFactory, class SharedFactory, 
    //         __EXECUTORS_REQUIRES(is_bulk_twoway_executor<Executor>::value)
    //        >
    //__AGENCY_ANNOTATION
    //auto bulk_twoway_execute(Function f, Shape shape, ResultFactory result_factory, SharedFactory shared_factory) const
    //{
    //  return base_executor_.bulk_twoway_execute(f, shape, result_factory, shared_factory);
    //}

    __agency_exec_check_disable__
    template<class Function, class Shape, class Future, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(is_bulk_then_executor<Executor>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, Shape shape, Future& fut, ResultFactory result_factory, Factories... shared_factories) const
    {
      return base_executor_.bulk_then_execute(f, shape, fut, result_factory, shared_factories...);
    }
};


} // end detail
} // end agency

