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
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/detail/has_require_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_query_member.hpp>
#include <agency/execution/executor/executor_traits/detail/has_static_query.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_bulk_twoway_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_then_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/is_single_twoway_executor.hpp>
#include <agency/execution/executor/customization_points/make_ready_future.hpp>
#include <agency/future.hpp>
#include <utility>


namespace agency
{
namespace detail
{


// this class wraps a base executor and forwards calls
// to execution functions to the base, if they exist
// the underlying executor may be stored as a value, or reference, depending on the template parameter
template<class Executor>
class basic_executor_adaptor
{
  private:
    // use Executor here instead of base_executor_type because Executor is allowed to be a reference
    Executor base_executor_;

    // decay the Executor type to strip off any reference for use in the __AGENCY_REQUIRES clauses below
    using base_executor_type = typename std::decay<Executor>::type;

  protected:
    __AGENCY_ANNOTATION
    const base_executor_type& base_executor() const
    {
      return base_executor_;
    }

  public:
    template<class T>
    using future = executor_future_t<base_executor_type,T>;

    using shape_type = executor_shape_t<base_executor_type>;

    // XXX need to publicize the other executor member typedefs such as index_type, etc.

    __agency_exec_check_disable__
    basic_executor_adaptor() = default;

    __agency_exec_check_disable__
    basic_executor_adaptor(const basic_executor_adaptor&) = default;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    basic_executor_adaptor(const base_executor_type& base) noexcept : base_executor_{base} {}


    // static query member function
    __agency_exec_check_disable__
    template<class Property,
             class E = base_executor_type,
             __AGENCY_REQUIRES(
               has_static_query<Property, E>::value
             )>
    __AGENCY_ANNOTATION
    constexpr static auto query(const Property&) ->
      decltype(Property::template static_query<E>())
    {
      return Property::template static_query<E>();
    }

    // non-static query member function
    __agency_exec_check_disable__
    template<class Property,
             class E = base_executor_type,
             __AGENCY_REQUIRES(
               !has_static_query<Property, E>::value and
               has_query_member<E,Property>::value
             )>
    __AGENCY_ANNOTATION
    constexpr auto query(const Property& p) const ->
      decltype(std::declval<const E>().query(p))
    {
      return base_executor().query(p);
    }

    __agency_exec_check_disable__
    template<class Property,
             class E = base_executor_type,
             __AGENCY_REQUIRES(
               has_require_member<E,Property>::value
             )>
    __AGENCY_ANNOTATION
    constexpr auto require(const Property& p) const ->
      decltype(std::declval<const E>().require(p))
    {
      return base_executor().require(p);
    }

    __agency_exec_check_disable__
    template<class Function,
             __AGENCY_REQUIRES(is_single_twoway_executor<base_executor_type>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<decay_t<Function>()>>
      twoway_execute(Function&& f) const
    {
      return base_executor_.twoway_execute(std::forward<Function>(f));
    }

    __agency_exec_check_disable__
    template<class Function, class Future,
             __AGENCY_REQUIRES(is_single_then_executor<base_executor_type>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_continuation_t<decay_t<Function>, Future>>
      then_execute(Function&& f, Future& fut) const
    {
      return base_executor_.then_execute(std::forward<Function>(f), fut);
    }

    __agency_exec_check_disable__
    template<class Function, class Shape, class ResultFactory, class... Factories, 
             __AGENCY_REQUIRES(is_bulk_twoway_executor<base_executor_type>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, Shape shape, ResultFactory result_factory, Factories... shared_factories) const
    {
      return base_executor_.bulk_twoway_execute(f, shape, result_factory, shared_factories...);
    }

    __agency_exec_check_disable__
    template<class Function, class Shape, class Future, class ResultFactory, class... Factories,
             __AGENCY_REQUIRES(is_bulk_then_executor<base_executor_type>::value)
            >
    __AGENCY_ANNOTATION
    future<result_of_t<ResultFactory()>>
      bulk_then_execute(Function f, Shape shape, Future& fut, ResultFactory result_factory, Factories... shared_factories) const
    {
      return base_executor_.bulk_then_execute(f, shape, fut, result_factory, shared_factories...);
    }
};


} // end detail
} // end agency

