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
#include <agency/execution/executor/executor_traits/detail/member_shape_type_or.hpp>
#include <agency/execution/executor/executor_traits/detail/member_index_type_or.hpp>
#include <agency/execution/executor/executor_traits/detail/executor_execution_depth_or.hpp>
#include <agency/execution/executor/executor_traits/detail/has_bulk_then_execute_member.hpp>
#include <utility>
#include <type_traits>
#include <future>


namespace agency
{
namespace detail
{
namespace is_bulk_then_executor_detail
{


template<class T, class IndexSequence>
struct is_bulk_then_executor_impl;

template<class T, size_t... Indices>
struct is_bulk_then_executor_impl<T, index_sequence<Indices...>>
{
  // executor properties
  using shape_type = member_shape_type_or_t<T,size_t>;
  using index_type = member_index_type_or_t<T,shape_type>;

  // types related to functions passed to .bulk_then_execute()
  using result_type = int;
  using predecessor_type = int;
  using predecessor_future_type = member_future_or_t<T,predecessor_type,std::future>;

  template<size_t>
  using shared_type = int;

  // the functions we'll pass to .bulk_then_execute() to test

  // XXX WAR nvcc 8.0 bug
  //using test_function = std::function<void(index_type, predecessor_type&, result_type&, shared_type<Indices>&...)>;
  //using test_result_factory = std::function<result_type()>;

  struct test_function
  {
    void operator()(index_type, predecessor_type&, result_type&, shared_type<Indices>&...);
  };

  struct test_result_factory
  {
    result_type operator()();
  };

  // XXX WAR nvcc 8.0 bug
  //template<size_t I>
  //using test_shared_factory = std::function<shared_type<I>()>;

  template<size_t I>
  struct test_shared_factory
  {
    shared_type<I> operator()();
  };

  using type = has_bulk_then_execute_member<
    T,
    test_function,
    shape_type,
    predecessor_future_type,
    test_result_factory,
    test_shared_factory<Indices>...
  >;
};


} // end is_bulk_then_executor_detail


template<class T>
using is_bulk_then_executor = typename is_bulk_then_executor_detail::is_bulk_then_executor_impl<
  T,
  make_index_sequence<
    executor_execution_depth_or<T>::value
  >
>::type;


} // end detail
} // end agency

