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
#include <agency/execution/executor/properties/always_blocking.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_is_always_blocking.hpp>


namespace agency
{
namespace detail
{


template<class Executor>
class always_blocking_executor : public basic_executor_adaptor<Executor>
{
  private:
    using super_t = basic_executor_adaptor<Executor>;

  public:
    template<class T>
    using future = typename super_t::template future<T>;

    __AGENCY_ANNOTATION
    always_blocking_executor(const Executor& ex) noexcept : super_t{ex} {}

    __AGENCY_ANNOTATION
    constexpr static bool query(always_blocking_t)
    {
      return true;
    }

    template<class Function,
             __AGENCY_REQUIRES(is_twoway_executor<super_t>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute(Function&& f) const
    {
      return twoway_execute_impl(std::forward<Function>(f));
    }

    template<class Function, class Future,
             __AGENCY_REQUIRES(is_then_executor<super_t>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_continuation_t<decay_t<Function>, Future>>
      then_execute(Function&& f, Future& fut) const
    {
      return then_execute_impl(std::forward<Function>(f), fut);
    }

    template<class Function, class Shape, class ResultFactory, class... Factories, 
             __AGENCY_REQUIRES(is_bulk_twoway_executor<super_t>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, Shape shape, ResultFactory result_factory, Factories... shared_factories) const
    {
      return bulk_twoway_execute_impl(f, shape, result_factory, shared_factories...);
    }

    template<class Function, class Shape, class Future, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(is_bulk_then_executor<super_t>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, Shape shape, Future& fut, ResultFactory result_factory, Factories... shared_factories) const
    {
      return bulk_then_execute_impl(f, shape, fut, result_factory, shared_factories...);
    }

  private:
    template<class Function,
             __AGENCY_REQUIRES(
               executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      return super_t::twoway_execute(std::forward<Function>(f));
    }

    template<class Function,
             __AGENCY_REQUIRES(
               !executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute_impl(Function&& f) const
    {
      auto future = super_t::twoway_execute(std::forward<Function>(f));

      future.wait();

      return future;
    }

    template<class Function, class Future,
             __AGENCY_REQUIRES(
               executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_continuation_t<decay_t<Function>, Future>>
      then_execute_impl(Function&& f, Future& fut) const
    {
      return super_t::then_execute(std::forward<Function>(f), fut);
    }

    template<class Function, class Future,
             __AGENCY_REQUIRES(
               !executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_continuation_t<decay_t<Function>, Future>>
      then_execute_impl(Function&& f, Future& fut) const
    {
      auto future = super_t::then_execute(std::forward<Function>(f), fut);

      future.wait();

      return future;
    }

    template<class Function, class Shape, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(
               executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_twoway_execute_impl(Function f, Shape shape, ResultFactory result_factory, Factories... shared_factories) const
    {
      return super_t::bulk_twoway_execute(f, shape, result_factory, shared_factories...);
    }

    __agency_exec_check_disable__
    template<class Function, class Shape, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(
               !executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_twoway_execute_impl(Function f, Shape shape, ResultFactory result_factory, Factories... shared_factories) const
    {
      auto future = super_t::bulk_twoway_execute(f, shape, result_factory, shared_factories...);

      future.wait();

      return future;
    }

    template<class Function, class Shape, class Future, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(
               executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute_impl(Function f, Shape shape, Future& fut, ResultFactory result_factory, Factories... shared_factories) const
    {
      return super_t::bulk_then_execute(f, shape, fut, result_factory, shared_factories...);
    }

    __agency_exec_check_disable__
    template<class Function, class Shape, class Future, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(
               !executor_is_always_blocking<super_t>::value
             )>
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute_impl(Function f, Shape shape, Future& fut, ResultFactory result_factory, Factories... shared_factories) const
    {
      auto future = super_t::bulk_then_execute(f, shape, fut, result_factory, shared_factories...);

      future.wait();

      return future;
    }
};


} // end detail
} // end agency

