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
#include <agency/execution/executor/executor_traits/detail/member_shape_type_or.hpp>
#include <agency/execution/executor/executor_traits/detail/member_index_type_or.hpp>
#include <agency/future/future_traits/is_future.hpp>
#include <type_traits>
#include <utility>


namespace agency
{
namespace detail
{


template<class T>
struct has_bulk_then_execute_member
{
  private:
    // define shorthands used below
    template<class U>
    using future = member_future_or_t<T, U, std::future>;
    using shape_type = member_shape_type_or_t<T, std::size_t>;
    using index_type = member_index_type_or_t<T, shape_type>;

    // dummy_functor & dummy_factory are used in bulk_then_execute_member_t below
    struct dummy_functor
    {
      // this overload is for non-void Predecessors
      template<class Predecessor, class Result, class Shared>
      void operator()(const index_type&, Predecessor&, Result&, Shared&) const;

      // this overload is for void Predecessors
      template<class Result, class Shared>
      void operator()(const index_type&, Result&, Shared&) const;
    };

    static int dummy_factory();

    template<class U, class Future,

             // XXX nomerge
             // XXX this should be changed to say std::declval<const U&>()
             //     once Agency's executors implement const execution functions as intended by P0443
             // first check for the existence of .bulk_then_execute()
             class Result = decltype(std::declval<U>().bulk_then_execute(std::declval<dummy_functor>(), std::declval<shape_type>(), std::declval<Future&>(), dummy_factory, dummy_factory)),

             // ensure that the Result is a Future type
             __AGENCY_REQUIRES(is_future<Result>::value)
            >
    static std::true_type test(int);

    template<class, class>
    static std::false_type test(...);

  public:
    // for safety, test T with both future<int> & future<void>
    static const bool value = decltype(test<T,future<int>>(0))::value and decltype(test<T,future<void>>(0))::value;
};


} // end detail
} // end agency

