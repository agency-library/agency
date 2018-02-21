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
#include <agency/execution/executor/detail/adaptors/bulk_then_executor.hpp>
#include <agency/execution/executor/detail/execution_functions/adaptations/then_execute_via_bulk_then_execute.hpp>


namespace agency
{
namespace detail
{


template<class BulkTwoWayExecutor, class Function, class Future,
         __AGENCY_REQUIRES(
           is_bulk_twoway_executor<BulkTwoWayExecutor>::value
         )>
__AGENCY_ANNOTATION
executor_future_t<BulkTwoWayExecutor, result_of_continuation_t<decay_t<Function>, Future>>
  then_execute_via_bulk_twoway_execute(const BulkTwoWayExecutor& exec, Function&& f, Future& fut)
{
  // XXX note that even though we've ensured that exec has .bulk_twoway_execute(),
  //     this implementation could actually call some other execution function exec has
  //     besides bulk_twoway_execute, so it's really not implemented correctly
  bulk_then_executor<BulkTwoWayExecutor> bulk_then_exec(exec);
  return detail::then_execute_via_bulk_then_execute(bulk_then_exec, std::forward<Function>(f), fut);
}


} // end detail
} // end agency

