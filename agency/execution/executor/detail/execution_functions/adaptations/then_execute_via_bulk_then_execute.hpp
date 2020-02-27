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
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/utility/bulk_then_execute_without_shared_parameters.hpp>
#include <agency/coordinate/detail/shape/shape_cast.hpp>
#include <agency/future/future_traits.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace then_execute_via_bulk_then_execute_detail
{


template<class Function>
struct then_execute_functor
{
  mutable Function f;

  // this overload of operator() handles the case when there is a non-void predecessor future
  template<class Index, class Predecessor, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Predecessor& predecessor, Result& result) const
  {
    result = invoke_and_return_unit_if_void_result(f, predecessor);
  }

  // this overload of operator() handles the case when there is a void predecessor future
  template<class Index, class Result>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result) const
  {
    result = invoke_and_return_unit_if_void_result(f);
  }
};


} // end then_execute_via_bulk_then_execute_detail


__agency_exec_check_disable__
template<class BulkThenExecutor, class Function, class Future,
         __AGENCY_REQUIRES(
           is_bulk_then_executor<BulkThenExecutor>::value
        )>
__AGENCY_ANNOTATION
executor_future_t<BulkThenExecutor, result_of_continuation_t<decay_t<Function>, Future>>
  then_execute_via_bulk_then_execute(const BulkThenExecutor& ex, Function&& f, Future& fut)
{
  using result_of_function = detail::result_of_continuation_t<Function,Future>;
  
  // if f returns void, then return a unit from bulk_then_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    detail::unit,
    result_of_function
  >::type;
  
  // XXX should really move f into this functor, but it's not clear how to make move-only
  //     parameters to CUDA kernels
  auto execute_me = then_execute_via_bulk_then_execute_detail::then_execute_functor<decay_t<Function>>{std::forward<Function>(f)};
  
  using shape_type = executor_shape_t<BulkThenExecutor>;
  
  // call bulk_then_execute_without_shared_parameters() to get an intermediate future
  auto intermediate_future = detail::bulk_then_execute_without_shared_parameters(
    ex,                                // the executor
    execute_me,                        // the functor to execute
    detail::shape_cast<shape_type>(1), // create only a single agent
    fut,                               // the predecessor argument to f
    detail::construct<result_type>()   // a factory for creating f's result
  );
  
  // cast the intermediate future into the right type of future for the result
  return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
}


} // end detail
} // end agency

