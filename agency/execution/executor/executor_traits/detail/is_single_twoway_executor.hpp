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
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/detail/member_future_or.hpp>
#include <utility>
#include <type_traits>


namespace agency
{
namespace detail
{
namespace is_single_twoway_executor_detail
{


template<class Executor, class Function>
struct has_single_twoway_execute_member_impl
{
  using result_type = result_of_t<Function()>;
  using expected_future_type = member_future_or_t<Executor,result_type,std::future>;

  template<class Executor1,
           class ReturnType = decltype(
             std::declval<const Executor1&>().twoway_execute(std::declval<Function>())
           ),
           class = typename std::enable_if<
             std::is_same<ReturnType,expected_future_type>::value
           >::type>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class Function>
using has_single_twoway_execute_member = typename has_single_twoway_execute_member_impl<Executor, Function>::type;


template<class T>
struct is_single_twoway_executor
{
  // the functions we'll pass to .twoway_execute() to test

  struct test_function
  {
    void operator()();
  };

  using type = typename has_single_twoway_execute_member<
    T,
    test_function
  >::type;
};


} // end is_single_twoway_executor_detail


template<class T>
using is_single_twoway_executor = typename is_single_twoway_executor_detail::is_single_twoway_executor<T>::type;


} // end detail
} // end agency

