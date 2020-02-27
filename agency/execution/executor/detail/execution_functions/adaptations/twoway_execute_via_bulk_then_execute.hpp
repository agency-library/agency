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
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/future/future_traits.hpp>
#include <agency/coordinate/detail/shape/shape_cast.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace twoway_execute_via_bulk_then_execute_detail
{


template<size_t>
using factory_returning_ignored_result = agency::detail::unit_factory;


struct twoway_execute_functor
{
  template<class Index, class Result, class SharedFunction, class... IgnoredArgs>
  __AGENCY_ANNOTATION
  void operator()(const Index&, Result& result, SharedFunction& shared_function, IgnoredArgs&...) const
  {
    result = detail::invoke_and_return_unit_if_void_result(shared_function);
  }
};


__agency_exec_check_disable__
template<size_t... Indices, class BulkThenExecutor, class Function>
__AGENCY_ANNOTATION
executor_future_t<BulkThenExecutor, result_of_t<decay_t<Function>()>>
  impl(index_sequence<Indices...>, const BulkThenExecutor& ex, Function&& f)
{
  using result_of_function = detail::result_of_t<Function()>;
  
  // if f returns void, then return a unit from bulk_then_execute()
  using result_type = typename std::conditional<
    std::is_void<result_of_function>::value,
    detail::unit,
    result_of_function
  >::type;
  
  using shape_type = executor_shape_t<BulkThenExecutor>;
  
  using void_future_type = executor_future_t<BulkThenExecutor, void>;
  
  // XXX we might want to actually allow the executor to participate here
  void_future_type predecessor = future_traits<void_future_type>::make_ready();
  
  auto intermediate_future = ex.bulk_then_execute(
    twoway_execute_functor(),                               // the functor to execute
    detail::shape_cast<shape_type>(1),                      // create only a single agent
    predecessor,                                            // an immediately ready predecessor future
    detail::construct<result_type>(),                       // a factory for creating f's result
    detail::make_moving_factory(std::forward<Function>(f)), // a factory to present f as the one shared parameter
    factory_returning_ignored_result<Indices>()...          // pass a factory for each inner level of execution hierarchy. the results of these factories will be ignored
  );
  
  // cast the intermediate future into the right type of future for the result
  return future_traits<decltype(intermediate_future)>::template cast<result_of_function>(intermediate_future);
}


} // end twoway_execute_via_bulk_then_execute_detail


template<class BulkThenExecutor, class Function,
         __AGENCY_REQUIRES(
           is_bulk_then_executor<BulkThenExecutor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<BulkThenExecutor, result_of_t<decay_t<Function>()>>
  twoway_execute_via_bulk_then_execute(const BulkThenExecutor& ex, Function&& f)
{
  using indices = make_index_sequence<executor_execution_depth<BulkThenExecutor>::value - 1>;
  return twoway_execute_via_bulk_then_execute_detail::impl(indices{}, ex, std::forward<Function>(f));
}


} // end detail
} // end agency


