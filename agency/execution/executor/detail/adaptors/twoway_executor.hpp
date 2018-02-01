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
#include <agency/detail/invoke.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/execution/executor/detail/utility/bulk_twoway_execute_with_one_shared_parameter.hpp>
#include <agency/execution/executor/detail/adaptors/basic_executor_adaptor.hpp>
#include <agency/execution/executor/detail/adaptors/adaptations/twoway_execute_via_bulk_then_execute.hpp>
#include <agency/execution/executor/detail/adaptors/adaptations/twoway_execute_via_bulk_twoway_execute.hpp>
#include <agency/execution/executor/detail/adaptors/adaptations/twoway_execute_via_then_execute.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/future.hpp>


namespace agency
{
namespace detail
{


template<class Executor>
class twoway_executor : public basic_executor_adaptor<Executor>
{
  private:
    using super_t = basic_executor_adaptor<Executor>;

  public:
    template<class T>
    using future = typename super_t::template future<T>;

    __AGENCY_ANNOTATION
    twoway_executor(const Executor& ex) noexcept : super_t{ex} {}

    template<class Function>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute(Function&& f) const
    {
      return twoway_execute_impl(std::forward<Function>(f));
    }

  private:
    __agency_exec_check_disable__
    template<class Function,
             __AGENCY_REQUIRES(is_twoway_executor<Executor>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      return super_t::base_executor().twoway_execute(std::forward<Function>(f));
    }

    // this case handles executors which have .then_execute() but not .twoway_execute()
    // XXX not really clear if we should prefer .bulk_twoway_execute() to calling .then_execute()
    // XXX one advantage of prioritizing an implementation using .then_execute() over .bulk_twoway_execute() is
    //     that no intermediate future is involved
    // XXX also, there's no weirdness involving move-only functions which .bulk_twoway_execute() would have trouble with
    __agency_exec_check_disable__
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<Executor>::value and
               is_then_executor<Executor>::value 
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      return detail::twoway_execute_via_then_execute(super_t::base_executor(), std::forward<Function>(f));
    }


    // this case handles executors which have .bulk_twoway_execute() but not .twoway_execute()
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<Executor>::value and
               !is_then_executor<Executor>::value and
               is_bulk_twoway_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      return detail::twoway_execute_via_bulk_twoway_execute(super_t::base_executor(), std::forward<Function>(f));
    }


    // this case handles executors which have .bulk_then_execute() but not .twoway_execute() or .bulk_twoway_execute()
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<Executor>::value and
               !is_then_executor<Executor>::value and
               !is_bulk_twoway_executor<Executor>::value and
               is_bulk_then_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      return detail::twoway_execute_via_bulk_then_execute(super_t::base_executor(), std::forward<Function>(f));
    }
};


} // end detail
} // end agency

