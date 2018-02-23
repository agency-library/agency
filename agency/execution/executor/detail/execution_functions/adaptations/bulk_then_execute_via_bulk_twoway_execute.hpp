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
#include <agency/execution/executor/executor_traits/executor_future.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/executor_execution_depth.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/future.hpp>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace bulk_then_execute_via_bulk_twoway_execute_impl
{


// this specialization of functor is for non-void SharedFuture
template<class Function, class SharedFuture,
         bool Enable = std::is_void<future_result_t<SharedFuture>>::value
        >
struct functor
{
  mutable Function f_;
  mutable SharedFuture fut_;

  using predecessor_type = future_result_t<SharedFuture>;

  __agency_exec_check_disable__
  ~functor() = default;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  functor(Function f, const SharedFuture& fut)
    : f_(f), fut_(fut)
  {}

  __agency_exec_check_disable__
  functor(const functor&) = default;

  __agency_exec_check_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index &idx, Args&... args) const ->
    decltype(f_(idx, const_cast<predecessor_type&>(fut_.get()),args...))
  {
    predecessor_type& predecessor = const_cast<predecessor_type&>(fut_.get());

    return f_(idx, predecessor, args...);
  }
};


// this specialization of functor is for void SharedFuture
template<class Function, class SharedFuture>
struct functor<Function,SharedFuture,true>
{
  mutable Function f_;
  mutable SharedFuture fut_;

  __agency_exec_check_disable__
  ~functor() = default;

  __agency_exec_check_disable__
  __AGENCY_ANNOTATION
  functor(Function f, const SharedFuture& fut)
    : f_(f), fut_(fut)
  {}

  __agency_exec_check_disable__
  functor(const functor&) = default;

  __agency_exec_check_disable__
  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  auto operator()(const Index &idx, Args&... args) const ->
    decltype(f_(idx, args...))
  {
    fut_.wait();

    return f_(idx, args...);
  }
};


template<class Function, class SharedFuture>
__AGENCY_ANNOTATION
functor<Function,SharedFuture> make_functor(Function f, const SharedFuture& shared_future)
{
  return functor<Function,SharedFuture>(f, shared_future);
}


} // end bulk_then_execute_via_bulk_twoway_execute_impl


__agency_exec_check_disable__
template<class BulkTwoWayExecutor, class Function, class Future, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(is_bulk_twoway_executor<BulkTwoWayExecutor>::value),
         __AGENCY_REQUIRES(executor_execution_depth<BulkTwoWayExecutor>::value == sizeof...(Factories))
         >
__AGENCY_ANNOTATION
executor_future_t<BulkTwoWayExecutor, result_of_t<ResultFactory()>>
  bulk_then_execute_via_bulk_twoway_execute(const BulkTwoWayExecutor& ex, Function f, executor_shape_t<BulkTwoWayExecutor> shape, Future& predecessor, ResultFactory result_factory, Factories... shared_factories)
{
  // XXX we may wish to allow the executor to participate in this sharing operation
  auto shared_predecessor_future = future_traits<Future>::share(predecessor);

  auto functor = bulk_then_execute_via_bulk_twoway_execute_impl::make_functor(f, shared_predecessor_future);

  return ex.bulk_twoway_execute(functor, shape, result_factory, shared_factories...);
}


} // end detail
} // end agency

