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
#include <agency/execution/executor/detail/adaptors/adaptations/then_execute_via_bulk_then_execute.hpp>
#include <agency/execution/executor/detail/adaptors/adaptations/then_execute_via_bulk_twoway_execute.hpp>
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>


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
    __AGENCY_ANNOTATION
    then_executor() = default;

    __AGENCY_ANNOTATION
    then_executor(const Executor& ex) noexcept : super_t{ex} {}

    template<class Function, class Future>
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
      then_execute(Function&& f, Future& fut) const
    {
      return then_execute_impl(std::forward<Function>(f), fut);
    }

  private:
    __agency_exec_check_disable__
    template<class Function, class Future,
             __AGENCY_REQUIRES(
               is_then_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
      then_execute_impl(Function&& f, Future& fut) const
    {
      return super_t::base_executor().then_execute(std::forward<Function>(f), fut);
    }

    template<class Function, class Future,
             __AGENCY_REQUIRES(
               !is_then_executor<Executor>::value and
               is_bulk_then_executor<Executor>::value
            )>
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
      then_execute_impl(Function&& f, Future& fut) const
    {
      return detail::then_execute_via_bulk_then_execute(super_t::base_executor(), std::forward<Function>(f), fut);
    }

    // XXX this is currently unimplemented
    //template<class Function, class Future,
    //         __AGENCY_REQUIRES(
    //           !is_then_executor<Executor>::value and
    //           !is_bulk_then_executor<Executor>::value
    //           is_twoway_executor<Executor>::value
    //         )>
    //__AGENCY_ANNOTATION
    //executor_future_t<E, result_of_continuation_t<decay_t<Function>, Future>>
    //  then_execute_impl(Function&& f, Future& fut) const;

    template<class Function, class Future,
             __AGENCY_REQUIRES(
               !is_then_executor<Executor>::value and
               !is_bulk_then_executor<Executor>::value and
               !is_twoway_executor<Executor>::value and
               is_bulk_twoway_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    executor_future_t<Executor, result_of_continuation_t<decay_t<Function>, Future>>
      then_execute_impl(Function&& f, Future& fut) const
    {
      return detail::then_execute_via_bulk_twoway_execute(super_t::base_executor(), std::forward<Function>(f), fut);
    }

    // XXX implement when Agency supports oneway executors
    //template<class Function, class T,
    //         __EXECUTORS_REQUIRES(
    //           !is_then_executor<Executor>::value
    //           and is_oneway_executor<Executor>::value
    //         )>
    //auto then_execute_impl(Function&& f, std::experimental::future<T>& fut) const;
};


} // end detail
} // end agency

