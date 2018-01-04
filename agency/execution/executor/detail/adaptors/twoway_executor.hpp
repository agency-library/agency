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
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/utility/bulk_async_execute_with_one_shared_parameter.hpp>
#include <agency/execution/executor/detail/adaptors/basic_executor_adaptor.hpp>
#include <agency/detail/shape_cast.hpp>


namespace agency
{
namespace detail
{


// this functor is used by twoway_executor below
// it is defined outside of the class so that it may be used
// as the template parameter of a CUDA kernel template
struct twoway_execute_functor
{
  template<class Index, class Result, class SharedFunction>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result, SharedFunction& shared_function) const
  {
    result = detail::invoke_and_return_unit_if_void_result(shared_function);
  }
};


template<class Executor>
class twoway_executor : public basic_executor_adaptor<Executor>
{
  private:
    using super_t = basic_executor_adaptor<Executor>;

  public:
    __AGENCY_ANNOTATION
    twoway_executor(const Executor& ex) noexcept : super_t{ex} {}

    template<class Function>
    __AGENCY_ANNOTATION
    executor_future_t<
      Executor,
      result_of_t<decay_t<Function>()>
    >
    twoway_execute(Function&& f) const
    {
      return twoway_execute_impl(std::forward<Function>(f));
    }

  private:
    template<class Function,
             __AGENCY_REQUIRES(is_twoway_executor<super_t>::value)
            >
    __AGENCY_ANNOTATION
    executor_future_t<
      Executor,
      result_of_t<decay_t<Function>()>
    >
    twoway_execute_impl(Function&& f) const
    {
      return super_t::twoway_execute(std::forward<Function>(f));
    }

    // this case handles executors which have .then_execute() but not .twoway_execute()
    // XXX not really clear if we should prefer .bulk_twoway_execute() to calling .then_execute()
    // XXX one advantage of prioritizing an implementation using .then_execute() over .bulk_twoway_execute() is
    //     that no intermediate future is involved
    // XXX also, there's no weirdness involving move-only functions which .bulk_twoway_execute() would have trouble with
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<super_t>::value and
               is_then_executor<super_t>::value 
             )>
    __AGENCY_ANNOTATION
    executor_future_t<
      Executor,
      result_of_t<decay_t<Function>()>
    >
    twoway_execute_impl(Function&& f) const
    {
      using void_future_type = executor_future_t<Executor,void>;

      // XXX should really allow the executor to participate here
      void_future_type ready_predecessor = future_traits<void_future_type>::make_ready();

      return super_t::then_execute(std::forward<Function>(f), ready_predecessor);
    }


    // this case handles executors which have no single-agent execution functions
    template<class Function,
             __AGENCY_REQUIRES(
               !is_twoway_executor<super_t>::value and
               !is_then_executor<super_t>::value and
               is_bulk_executor<Executor>::value
             )>
    __AGENCY_ANNOTATION
    executor_future_t<
      Executor,
      result_of_t<decay_t<Function>()>
    >
    twoway_execute_impl(Function&& f) const
    {
      using result_of_function = detail::result_of_t<Function()>;

      // if f returns void, then return a unit from bulk_async_execute()
      using result_type = typename std::conditional<
        std::is_void<result_of_function>::value,
        detail::unit,
        result_of_function
      >::type;

      using shape_type = executor_shape_t<Executor>;

      // XXX eliminate this and just call base_executor() below once Agency's
      //     executors implement shallow-constness correctly
      Executor& exec = super_t::base_executor();

      auto intermediate_future = agency::detail::bulk_async_execute_with_one_shared_parameter(
        exec,                                                  // the executor
        twoway_execute_functor(),                              // the functor to execute
        detail::shape_cast<shape_type>(1),                     // create only a single agent
        detail::construct<result_type>(),                      // a factory for creating f's result
        detail::make_moving_factory(std::forward<Function>(f)) // a factory to present f as the one shared parameter
      );

      // cast the intermediate future into the right type of future for the result
      return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
    }
};


} // end detail
} // end agency
